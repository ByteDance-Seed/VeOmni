# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ExtraParallel: streaming weight load and gradient-norm clipping.

Covers four layers in one file:
- Stream load CPU unit tests (``LaunchSpec(device="cpu")``): per-expert -> fused stacking
  (Qwen3-MoE, DeepSeek-V3/V4, GLM-5, Qwen3-Omni-MoE) and ``dim0_rowwise_companion``.
- 2-accelerator non-PEFT MoE load matrix (``LaunchSpec(nproc=2)``): bit-identical load
  across plain, rank0-broadcast, and streaming for merged / per-expert checkpoints.
- 2-accelerator PEFT MoE-LoRA stream load (``LaunchSpec(nproc=2)``): bit-identical
  streaming vs broadcast load for independent and shared MoE-LoRA adapters.
- ExtraParallel gradient-norm clipping: 4-accelerator FSDP2 and 8-accelerator HSDP
  grad-norm parity (including ``extra_parallel_only``) plus CPU inf-norm/reduction tests.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from packaging.version import Version
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Shard

import veomni.models.checkpoint.weights as checkpoint_weights
from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from veomni.arguments import MixedPrecisionConfig, ModelArguments
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    use_parallel_state,
)
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.checkpoint.convert import ConvertedCheckpointTensor
from veomni.models.transformers.deepseek_v3.checkpoint_tensor_converter import DeepseekV3CheckpointTensorConverter
from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import DeepseekV4CheckpointTensorConverter
from veomni.models.transformers.glm_moe_dsa.checkpoint_tensor_converter import GlmMoeDsaCheckpointTensorConverter
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter
from veomni.models.transformers.qwen3_omni_moe.checkpoint_tensor_converter import (
    Qwen3OmniMoeCheckpointTensorConverter,
)
from veomni.optim import build_optimizer
from veomni.utils import helper
from veomni.utils.device import get_device_id, get_device_type, get_dist_comm_backend, get_torch_device


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


NUM_EXPERTS, HIDDEN, INTERMEDIATE, EP_SIZE = 4, 3, 2, 2
PER_EXPERT = "mtp.layers.0.mlp.experts.{expert}.{proj}.weight"


class _Experts(nn.Module):
    def __init__(self, num_local: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_local, 2 * INTERMEDIATE, HIDDEN))
        self.down_proj = nn.Parameter(torch.empty(num_local, HIDDEN, INTERMEDIATE))


def _layer(num_local: int) -> nn.Module:
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = _Experts(num_local)
    return layer


class _MtpMoeModel(nn.Module):
    converter_cls = Qwen3MoeCheckpointTensorConverter

    def __init__(self, with_mtp: bool = True) -> None:
        super().__init__()
        num_local = NUM_EXPERTS // EP_SIZE
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_layer(num_local)])
        if with_mtp:
            self.mtp = nn.Module()
            self.mtp.layers = nn.ModuleList([_layer(num_local)])
        self.lm_head = nn.Linear(HIDDEN, 5, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(
            extra_parallel_plan={
                "ep": {
                    f"{root}.layers.*.mlp.experts.{proj}": Shard(0)
                    for root in (("model", "mtp") if hasattr(self, "mtp") else ("model",))
                    for proj in ("gate_up_proj", "down_proj")
                }
            }
        )

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return model.converter_cls(num_experts=NUM_EXPERTS)


class _WholeSetOnlyConverter:
    """A per-expert fusion converter without the expert-streaming capabilities."""

    def __init__(self, num_experts: int) -> None:
        self._inner = Qwen3MoeCheckpointTensorConverter(num_experts)
        self.can_handle = self._inner.can_handle
        self.convert = self._inner.convert
        self.finalize = self._inner.finalize


def _write_checkpoint(path: Path) -> dict:
    generator = torch.Generator().manual_seed(0)
    state = {
        "lm_head.weight": torch.randn(5, HIDDEN, generator=generator),
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator
        ),
        "model.layers.0.mlp.experts.down_proj": torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator),
    }
    for expert in range(NUM_EXPERTS):
        for proj, shape in (
            ("gate_proj", (INTERMEDIATE, HIDDEN)),
            ("up_proj", (INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, INTERMEDIATE)),
        ):
            state[PER_EXPERT.format(expert=expert, proj=proj)] = torch.randn(*shape, generator=generator)
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path / "model.safetensors"))
    return state


def _load_as_rank(monkeypatch, weights_path: Path, ep_rank: int, model: nn.Module) -> list:
    """Stream ``weights_path`` into ``model`` as ``ep_rank``; return the keys read whole."""
    parallel_state = SimpleNamespace(
        extra_parallel_enabled=lambda name: True,
        extra_parallel_sizes={"ep": EP_SIZE},
        extra_parallel_rank=lambda name: ep_rank,
    )
    monkeypatch.setattr(checkpoint_weights, "get_parallel_state", lambda: parallel_state)

    read_keys = []

    class _RecordingSafeOpen:
        def __init__(self, *args, **kwargs):
            self._handle = safe_open(*args, **kwargs)

        def __enter__(self):
            handle = self._handle.__enter__()
            outer = self

            class _Proxy:
                def get_tensor(self, key):
                    read_keys.append(key)
                    return handle.get_tensor(key)

                def __getattr__(self, attr):
                    return getattr(handle, attr)

            outer._proxy = _Proxy()
            return outer._proxy

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    monkeypatch.setattr(checkpoint_weights, "safe_open", _RecordingSafeOpen)
    checkpoint_weights.load_model_weights_ep_sharded(model, str(weights_path), init_device="cpu")
    return read_keys


def _meta_model(converter_cls=Qwen3MoeCheckpointTensorConverter, with_mtp: bool = True) -> nn.Module:
    with torch.device("meta"):
        model = _MtpMoeModel(with_mtp)
    model.converter_cls = converter_cls
    return model


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_per_expert_mtp_experts_stream_into_the_local_fused_slice(monkeypatch, tmp_path, ep_rank):
    state = _write_checkpoint(tmp_path)
    model = _meta_model()

    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    num_local = NUM_EXPERTS // EP_SIZE
    local = range(ep_rank * num_local, (ep_rank + 1) * num_local)
    want_gate_up = torch.stack(
        [
            torch.cat(
                [
                    state[PER_EXPERT.format(expert=expert, proj="gate_proj")],
                    state[PER_EXPERT.format(expert=expert, proj="up_proj")],
                ]
            )
            for expert in local
        ]
    )
    want_down = torch.stack([state[PER_EXPERT.format(expert=expert, proj="down_proj")] for expert in local])
    mtp_experts = model.mtp.layers[0].mlp.experts
    torch.testing.assert_close(mtp_experts.gate_up_proj.data, want_gate_up, atol=0, rtol=0)
    torch.testing.assert_close(mtp_experts.down_proj.data, want_down, atol=0, rtol=0)
    trunk_experts = model.model.layers[0].mlp.experts
    rows = slice(local.start, local.stop)
    torch.testing.assert_close(
        trunk_experts.gate_up_proj.data, state["model.layers.0.mlp.experts.gate_up_proj"][rows], atol=0, rtol=0
    )
    torch.testing.assert_close(
        trunk_experts.down_proj.data, state["model.layers.0.mlp.experts.down_proj"][rows], atol=0, rtol=0
    )

    read_experts = {int(key.split(".")[5]) for key in read_keys if key.startswith("mtp.")}
    assert read_experts == set(local)


def test_a_fusion_converter_without_expert_streaming_still_bails(monkeypatch, tmp_path):
    """Without ``fused_expert_target`` / ``for_expert_range`` the per-expert keys
    would be dropped as unexpected, leaving the fused experts uninitialised."""
    _write_checkpoint(tmp_path)

    with pytest.raises(NotImplementedError, match="cannot stream it per expert"):
        _load_as_rank(monkeypatch, tmp_path, 0, _meta_model(_WholeSetOnlyConverter))


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_a_checkpoint_missing_one_ranks_whole_expert_range_raises_on_every_rank(monkeypatch, tmp_path, ep_rank):
    """Rank 1 would buffer nothing for experts 2-3, so its ``finalize`` alone cannot
    flag the gap; the whole-tensor loader rejects this checkpoint, and so must this one."""
    state = _write_checkpoint(tmp_path)
    for expert in range(NUM_EXPERTS // EP_SIZE, NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            del state[PER_EXPERT.format(expert=expert, proj=proj)]
    save_file(state, str(tmp_path / "model.safetensors"))

    with pytest.raises(
        RuntimeError, match=r"incomplete checkpoint detected .*missing per-expert keys for experts \[2, 3\]"
    ):
        _load_as_rank(monkeypatch, tmp_path, ep_rank, _meta_model())


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_an_expert_missing_one_fused_source_key_raises_on_every_rank(monkeypatch, tmp_path, ep_rank):
    """Expert 3 keeps ``up_proj`` but loses ``gate_proj``: only rank 1's ``finalize``
    would see the unpaired buffer, so the up-front scan must catch it for rank 0 too."""
    state = _write_checkpoint(tmp_path)
    del state[PER_EXPERT.format(expert=NUM_EXPERTS - 1, proj="gate_proj")]
    save_file(state, str(tmp_path / "model.safetensors"))

    with pytest.raises(RuntimeError, match=rf"experts \[{NUM_EXPERTS - 1}\] have fewer than 2 source keys"):
        _load_as_rank(monkeypatch, tmp_path, ep_rank, _meta_model())


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_a_fused_source_key_missing_for_every_expert_raises_on_every_rank(monkeypatch, tmp_path, ep_rank):
    """Uniform per-expert counts pass the up-front scan; each rank then stacks the
    surviving ``up_proj`` with no ``gate_proj`` to pair, and its ``finalize`` raises."""
    state = _write_checkpoint(tmp_path)
    for expert in range(NUM_EXPERTS):
        del state[PER_EXPERT.format(expert=expert, proj="gate_proj")]
    save_file(state, str(tmp_path / "model.safetensors"))

    with pytest.raises(RuntimeError, match="missing gate/up pair"):
        _load_as_rank(monkeypatch, tmp_path, ep_rank, _meta_model())


def test_a_checkpoint_with_more_experts_than_the_model_raises(monkeypatch, tmp_path):
    state = _write_checkpoint(tmp_path)
    for proj, shape in (("gate_proj", (INTERMEDIATE, HIDDEN)), ("up_proj", (INTERMEDIATE, HIDDEN))):
        state[PER_EXPERT.format(expert=NUM_EXPERTS, proj=proj)] = torch.zeros(*shape)
    save_file(state, str(tmp_path / "model.safetensors"))

    with pytest.raises(RuntimeError, match=rf"\({NUM_EXPERTS} experts\): unexpected experts \[{NUM_EXPERTS}\]\.$"):
        _load_as_rank(monkeypatch, tmp_path, 0, _meta_model())


class _PerExpertTrunkModel(nn.Module):
    """A one-layer MoE trunk under ``root`` whose checkpoint stores every expert per expert."""

    def __init__(self, root: str, converter_cls) -> None:
        super().__init__()
        parent = self
        for part in root.split("."):
            parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.layers = nn.ModuleList([_layer(NUM_EXPERTS // EP_SIZE)])
        self.root = root
        self.converter_cls = converter_cls
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(
            extra_parallel_plan={
                "ep": {f"{self.root}.layers.*.mlp.experts.{proj}": Shard(0) for proj in ("gate_up_proj", "down_proj")}
            }
        )

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return model.converter_cls(num_experts=NUM_EXPERTS)


@pytest.mark.parametrize(
    "converter_cls, built, unbuilt",
    [
        pytest.param(Qwen3MoeCheckpointTensorConverter, "model.layers.0", None, id="qwen3_moe"),
        # DeepSeek-V3 ships its MTP layer as one extra trunk index the model does not build.
        pytest.param(DeepseekV3CheckpointTensorConverter, "model.layers.0", "model.layers.1", id="deepseek_v3"),
        # So does GLM-5 (``model.layers.78``).
        pytest.param(GlmMoeDsaCheckpointTensorConverter, "model.layers.0", "model.layers.1", id="glm_moe_dsa"),
        # VeOmni builds Qwen3-Omni-MoE with has_talker=False.
        pytest.param(
            Qwen3OmniMoeCheckpointTensorConverter,
            "thinker.model.layers.0",
            "talker.model.layers.0",
            id="qwen3_omni_moe",
        ),
    ],
)
def test_an_all_per_expert_trunk_streams_into_the_local_fused_slice(
    monkeypatch, tmp_path, converter_cls, built, unbuilt
):
    generator = torch.Generator().manual_seed(0)
    state = {}
    for layer in filter(None, (built, unbuilt)):
        for expert in range(NUM_EXPERTS):
            for proj, shape in (
                ("gate_proj", (INTERMEDIATE, HIDDEN)),
                ("up_proj", (INTERMEDIATE, HIDDEN)),
                ("down_proj", (HIDDEN, INTERMEDIATE)),
            ):
                state[f"{layer}.mlp.experts.{expert}.{proj}.weight"] = torch.randn(*shape, generator=generator)
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(tmp_path / "model.safetensors"))

    ep_rank = 1
    with torch.device("meta"):
        model = _PerExpertTrunkModel(built.rsplit(".layers.", 1)[0], converter_cls)
    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    num_local = NUM_EXPERTS // EP_SIZE
    local = range(ep_rank * num_local, (ep_rank + 1) * num_local)
    key = f"{built}.mlp.experts.{{expert}}.{{proj}}.weight"
    experts = model.get_submodule(f"{built}.mlp.experts")
    want_gate_up = torch.stack(
        [
            torch.cat([state[key.format(expert=e, proj="gate_proj")], state[key.format(expert=e, proj="up_proj")]])
            for e in local
        ]
    )
    want_down = torch.stack([state[key.format(expert=e, proj="down_proj")] for e in local])
    torch.testing.assert_close(experts.gate_up_proj.data, want_gate_up, atol=0, rtol=0)
    torch.testing.assert_close(experts.down_proj.data, want_down, atol=0, rtol=0)
    assert {int(k.split(".experts.")[1].split(".")[0]) for k in read_keys} == set(local)
    if unbuilt is not None:
        assert not any(k.startswith(unbuilt + ".") for k in read_keys)


def test_per_expert_keys_of_an_unbuilt_module_are_skipped_unread(monkeypatch, tmp_path):
    """With MTP off the model has no ``mtp`` experts; the checkpoint's per-expert MTP
    keys are unexpected, as on the whole-tensor loader, and never read."""
    state = _write_checkpoint(tmp_path)
    model = _meta_model(with_mtp=False)

    read_keys = _load_as_rank(monkeypatch, tmp_path, 1, model)

    assert not any(key.startswith("mtp.") for key in read_keys)
    num_local = NUM_EXPERTS // EP_SIZE
    torch.testing.assert_close(
        model.model.layers[0].mlp.experts.down_proj.data,
        state["model.layers.0.mlp.experts.down_proj"][num_local:],
        atol=0,
        rtol=0,
    )


class _DeepseekV4StreamModel(nn.Module):
    """One DSV4-named MoE layer plus dense ``q_a_proj`` / ``lm_head``.

    The checkpoint uses DeepSeek's inference names (``layers.0.ffn.experts.{j}.w{1,2,3}``,
    ``layers.0.attn.wq_a``, ``head``), so every key goes through the converter's rename.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        layer = _layer(NUM_EXPERTS // EP_SIZE)
        layer.self_attn = nn.Module()
        layer.self_attn.q_a_proj = nn.Linear(HIDDEN, Q_A_DIM, bias=False)
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(HIDDEN, 5, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(
            extra_parallel_plan={
                "ep": {f"model.layers.*.mlp.experts.{proj}": Shard(0) for proj in ("gate_up_proj", "down_proj")}
            }
        )

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)


Q_A_DIM = 4
_DSV4_EXPERT_SHAPES = (("w1", (INTERMEDIATE, HIDDEN)), ("w3", (INTERMEDIATE, HIDDEN)), ("w2", (HIDDEN, INTERMEDIATE)))


def _dsv4_local_experts(ep_rank: int) -> range:
    num_local = NUM_EXPERTS // EP_SIZE
    return range(ep_rank * num_local, (ep_rank + 1) * num_local)


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_deepseek_v4_w_keys_stream_into_the_local_fused_slice(monkeypatch, tmp_path, ep_rank):
    """BF16 ``w1/w2/w3`` experts stream per rank; renamed dense keys are read whole."""
    generator = torch.Generator().manual_seed(0)
    state = {
        "head.weight": torch.randn(5, HIDDEN, generator=generator),
        "layers.0.attn.wq_a.weight": torch.randn(Q_A_DIM, HIDDEN, generator=generator),
    }
    for expert in range(NUM_EXPERTS):
        for proj, shape in _DSV4_EXPERT_SHAPES:
            state[f"layers.0.ffn.experts.{expert}.{proj}.weight"] = torch.randn(*shape, generator=generator)
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(tmp_path / "model.safetensors"))

    with torch.device("meta"):
        model = _DeepseekV4StreamModel()
    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    local = _dsv4_local_experts(ep_rank)
    expert = "layers.0.ffn.experts.{}.{}.weight"
    want_gate_up = torch.stack(
        [torch.cat([state[expert.format(e, "w1")], state[expert.format(e, "w3")]]) for e in local]
    )
    want_down = torch.stack([state[expert.format(e, "w2")] for e in local])
    layer = model.model.layers[0]
    torch.testing.assert_close(layer.mlp.experts.gate_up_proj.data, want_gate_up, atol=0, rtol=0)
    torch.testing.assert_close(layer.mlp.experts.down_proj.data, want_down, atol=0, rtol=0)
    torch.testing.assert_close(
        layer.self_attn.q_a_proj.weight.data, state["layers.0.attn.wq_a.weight"], atol=0, rtol=0
    )
    torch.testing.assert_close(model.lm_head.weight.data, state["head.weight"], atol=0, rtol=0)
    assert {int(k.split(".experts.")[1].split(".")[0]) for k in read_keys if ".ffn.experts." in k} == set(local)


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_deepseek_v4_fp8_weights_and_scales_stream_with_their_owner(monkeypatch, tmp_path, ep_rank):
    """Each rank pairs and dequantizes weight+scale for its own experts only; the dense
    FP8 ``wq_a`` pair is dequantized on every rank."""
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")

    scales = {"w1": 2.0, "w2": 3.0, "w3": 4.0}
    state = {
        "head.weight": torch.full((5, HIDDEN), 0.25),
        "layers.0.attn.wq_a.weight": torch.full((Q_A_DIM, HIDDEN), 1.5).to(fp8_dtype),
        "layers.0.attn.wq_a.scale": torch.tensor([[5.0]]),
    }
    for expert in range(NUM_EXPERTS):
        for proj, shape in _DSV4_EXPERT_SHAPES:
            stem = f"layers.0.ffn.experts.{expert}.{proj}"
            state[f"{stem}.weight"] = torch.full(shape, float(expert + 1)).to(fp8_dtype)
            state[f"{stem}.scale"] = torch.tensor([[scales[proj]]])
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(tmp_path / "model.safetensors"))

    with torch.device("meta"):
        model = _DeepseekV4StreamModel()
    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    local = _dsv4_local_experts(ep_rank)
    experts = model.model.layers[0].mlp.experts
    for row, e in enumerate(local):
        want = {proj: torch.full(shape, float(e + 1)) * scales[proj] for proj, shape in _DSV4_EXPERT_SHAPES}
        torch.testing.assert_close(experts.gate_up_proj.data[row], torch.cat([want["w1"], want["w3"]]), atol=0, rtol=0)
        torch.testing.assert_close(experts.down_proj.data[row], want["w2"], atol=0, rtol=0)
    q_a = model.model.layers[0].self_attn.q_a_proj.weight.data
    torch.testing.assert_close(q_a, torch.full((Q_A_DIM, HIDDEN), 7.5), atol=0, rtol=0)

    assert {k for k in read_keys if ".ffn.experts." in k} == {
        f"layers.0.ffn.experts.{e}.{proj}.{part}" for e in local for proj in scales for part in ("weight", "scale")
    }


def _save_split_by_suffix(path: Path, state: dict, scales_first: bool) -> None:
    """Write ``.scale`` tensors and everything else to two shards, scales in the first one or the last one."""
    scales = {k: v for k, v in state.items() if k.endswith(".scale")}
    rest = {k: v for k, v in state.items() if not k.endswith(".scale")}
    first, second = (scales, rest) if scales_first else (rest, scales)
    path.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i, shard in enumerate((first, second), start=1):
        fname = f"model-0000{i}-of-00002.safetensors"
        save_file(shard, str(path / fname))
        weight_map.update(dict.fromkeys(shard, fname))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
@pytest.mark.parametrize("scales_first", (True, False), ids=("scales_first", "scales_last"))
@pytest.mark.parametrize(
    "q_a_stem", ("layers.0.attn.wq_a", "model.layers.0.self_attn.q_a_proj"), ids=("raw_dense", "hf_named_dense")
)
def test_deepseek_v4_fp8_pairs_split_across_shards(monkeypatch, tmp_path, ep_rank, scales_first, q_a_stem):
    """Weight and scale may sit in different shards, in either order. An HF-named FP8
    dense weight is claimed only by dtype (``can_handle_tensor``), not by name."""
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")

    state = {
        "head.weight": torch.full((5, HIDDEN), 0.25),
        f"{q_a_stem}.weight": torch.full((Q_A_DIM, HIDDEN), 1.5).to(fp8_dtype),
        f"{q_a_stem}.scale": torch.tensor([[5.0]]),
    }
    for expert in range(NUM_EXPERTS):
        for proj, shape in _DSV4_EXPERT_SHAPES:
            stem = f"layers.0.ffn.experts.{expert}.{proj}"
            state[f"{stem}.weight"] = torch.full(shape, float(expert + 1)).to(fp8_dtype)
            state[f"{stem}.scale"] = torch.tensor([[2.0]])
    _save_split_by_suffix(tmp_path, state, scales_first)

    with torch.device("meta"):
        model = _DeepseekV4StreamModel()
    _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    experts = model.model.layers[0].mlp.experts
    for row, e in enumerate(_dsv4_local_experts(ep_rank)):
        torch.testing.assert_close(experts.down_proj.data[row], torch.full((HIDDEN, INTERMEDIATE), 2.0 * (e + 1)))
    q_a = model.model.layers[0].self_attn.q_a_proj.weight.data
    torch.testing.assert_close(q_a, torch.full((Q_A_DIM, HIDDEN), 7.5), atol=0, rtol=0)


def test_deepseek_v4_orphan_scale_loads_like_the_whole_tensor_loader(monkeypatch, tmp_path):
    """A dequantized BF16 weight whose ``.scale`` stayed behind: the converter returns the
    scale at ``finalize``, which is dropped as unexpected rather than failing the load."""
    q_a = torch.full((Q_A_DIM, HIDDEN), 1.5)
    state = {
        "head.weight": torch.full((5, HIDDEN), 0.25),
        "layers.0.attn.wq_a.weight": q_a,
        "layers.0.attn.wq_a.scale": torch.tensor([[5.0]]),
    }
    for expert in range(NUM_EXPERTS):
        for proj, shape in _DSV4_EXPERT_SHAPES:
            state[f"layers.0.ffn.experts.{expert}.{proj}.weight"] = torch.zeros(shape)
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(tmp_path / "model.safetensors"))

    with torch.device("meta"):
        model = _DeepseekV4StreamModel()
    _load_as_rank(monkeypatch, tmp_path, 0, model)

    torch.testing.assert_close(model.model.layers[0].self_attn.q_a_proj.weight.data, q_a, atol=0, rtol=0)


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_deepseek_v4_orphan_expert_scales_beside_bf16_experts_are_dropped(monkeypatch, tmp_path, ep_rank):
    """Expert ``.scale`` keys left beside dequantized BF16 expert weights: each rank's
    expert-range converter returns its orphan scales at ``finalize``, which are dropped
    as unexpected (as on the whole-tensor loader) instead of failing the load."""
    generator = torch.Generator().manual_seed(0)
    state = {
        "head.weight": torch.randn(5, HIDDEN, generator=generator),
        "layers.0.attn.wq_a.weight": torch.randn(Q_A_DIM, HIDDEN, generator=generator),
    }
    for expert in range(NUM_EXPERTS):
        for proj, shape in _DSV4_EXPERT_SHAPES:
            stem = f"layers.0.ffn.experts.{expert}.{proj}"
            state[f"{stem}.weight"] = torch.randn(*shape, generator=generator)
            state[f"{stem}.scale"] = torch.tensor([[2.0]])
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(tmp_path / "model.safetensors"))

    with torch.device("meta"):
        model = _DeepseekV4StreamModel()
    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    local = _dsv4_local_experts(ep_rank)
    expert = "layers.0.ffn.experts.{}.{}.weight"
    want_down = torch.stack([state[expert.format(e, "w2")] for e in local])
    torch.testing.assert_close(model.model.layers[0].mlp.experts.down_proj.data, want_down, atol=0, rtol=0)
    assert {int(k.split(".experts.")[1].split(".")[0]) for k in read_keys if ".ffn.experts." in k} == set(local)


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_dim0_rowwise_companion_streams_sharded_with_zero_pad_and_dense(monkeypatch, tmp_path, ep_rank):
    """``dim0_rowwise_companion`` + ``convert_rows`` streams matching row slices for sharded tables
    (zero-filling padded tail rows) and converts non-sharded keys whole without loading companion keys separately."""

    class _RowScaleConverter:
        def can_handle(self, name: str) -> bool:
            return name.endswith(".scale")

        def convert(self, name: str, tensor: torch.Tensor):
            return None

        def dim0_rowwise_companion(self, name: str):
            return f"{name}.scale" if name.endswith(".weight") else None

        def convert_rows(self, name: str, rows: torch.Tensor, companion_rows: torch.Tensor) -> torch.Tensor:
            return rows.float() * companion_rows.float()

        def finalize(self):
            return []

    class _CompanionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 6 logical rows padded to 8 across EP_SIZE=2 ranks -> 4 local rows per rank.
            self.table = nn.Embedding(4, HIDDEN)
            self.dense = nn.Linear(HIDDEN, 3, bias=False)
            self.config = SimpleNamespace(tie_word_embeddings=False)

        def get_parallel_plan(self) -> ParallelPlan:
            return ParallelPlan(extra_parallel_plan={"ep": {"table.weight": Shard(0)}})

        @staticmethod
        def _create_checkpoint_tensor_converter(_model):
            return _RowScaleConverter()

    table_w = torch.arange(6 * HIDDEN, dtype=torch.float32).view(6, HIDDEN)
    table_s = torch.arange(1, 7, dtype=torch.float32).view(6, 1)
    dense_w = torch.arange(3 * HIDDEN, dtype=torch.float32).view(3, HIDDEN)
    dense_s = torch.tensor([[2.0], [3.0], [4.0]])
    save_file(
        {
            "table.weight": table_w,
            "table.weight.scale": table_s,
            "dense.weight": dense_w,
            "dense.weight.scale": dense_s,
        },
        str(tmp_path / "model.safetensors"),
    )

    with torch.device("meta"):
        model = _CompanionModel()
    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model)

    # Sharded table is never read whole, and companion keys are never dispatched standalone.
    assert "table.weight" not in read_keys
    assert "table.weight.scale" not in read_keys

    expected_full_table = torch.cat([table_w * table_s, torch.zeros(2, HIDDEN)], dim=0)
    local_expected = expected_full_table[ep_rank * 4 : (ep_rank + 1) * 4]
    torch.testing.assert_close(model.table.weight.data, local_expected, atol=0, rtol=0)
    torch.testing.assert_close(model.dense.weight.data, dense_w * dense_s, atol=0, rtol=0)


# ======================================================================
# 2-accelerator non-PEFT MoE load matrix (plain / rank0 / stream load)
# ======================================================================

_CI_LAUNCH_2 = LaunchSpec(nproc=2)
_MATRIX_E, _MATRIX_OUT, _MATRIX_IN = 8, 6, 4
_MATRIX_DENSE_OUT, _MATRIX_DENSE_IN = 3, 4
_MATRIX_VOCAB, _MATRIX_EMB = 10, 4


class _PerExpertFuseConverter:
    """Per-expert -> fused converter without expert-streaming hooks (streaming must bail)."""

    _PREFIX = "moe.experts."

    def __init__(self) -> None:
        self._buf: dict[int, torch.Tensor] = {}

    def can_handle(self, name: str) -> bool:
        return name.startswith(self._PREFIX) and name[len(self._PREFIX) :].isdigit()

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        idx = int(name[len(self._PREFIX) :])
        self._buf[idx] = tensor.clone()
        return self._flush() if len(self._buf) == _MATRIX_E else None

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        return [self._flush()] if self._buf else []

    def _flush(self) -> ConvertedCheckpointTensor:
        fused = torch.stack([self._buf[i] for i in sorted(self._buf)], dim=0)
        self._buf.clear()
        return ConvertedCheckpointTensor(name="moe.experts", tensor=fused)

    def is_dim0_zero_pad(self, name: str) -> bool:
        return False


class _StreamablePerExpertFuseConverter(_PerExpertFuseConverter):
    """Per-expert -> fused converter with expert-streaming hooks."""

    def __init__(self, num_experts: int = _MATRIX_E, offset: int = 0) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.offset = offset

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        idx = int(name[len(self._PREFIX) :]) - self.offset
        assert 0 <= idx < self.num_experts, f"{name} is outside [{self.offset}, {self.offset + self.num_experts})"
        self._buf[idx] = tensor.clone()
        return self._flush() if len(self._buf) == self.num_experts else None

    def fused_expert_target(self, name: str) -> tuple[str, int]:
        return "moe.experts", int(name[len(self._PREFIX) :])

    def for_expert_range(self, start: int, num_local: int) -> _StreamablePerExpertFuseConverter:
        return _StreamablePerExpertFuseConverter(num_local, self.offset + start)


class _FakeExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.Parameter(torch.empty(_MATRIX_E, _MATRIX_OUT, _MATRIX_IN))


class FakeMoeModel(nn.Module):
    _no_split_modules = ["_FakeExperts"]

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.empty(_MATRIX_VOCAB, _MATRIX_EMB))
        self.linear = nn.Linear(_MATRIX_DENSE_IN, _MATRIX_DENSE_OUT, bias=True)
        self.moe = _FakeExperts()
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def init_weights(self) -> None:
        with torch.no_grad():
            self.embed.normal_()
            nn.init.xavier_uniform_(self.linear.weight)
            self.linear.bias.fill_(0.0)
            nn.init.xavier_uniform_(self.moe.experts)

    def get_parallel_plan(self) -> ParallelPlan:
        plan = ParallelPlan(extra_parallel_plan={"ep": {"moe.experts": Shard(0)}})
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"moe"}}
        return plan

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return _PerExpertFuseConverter()


class FakeStreamableMoeModel(FakeMoeModel):
    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return _StreamablePerExpertFuseConverter()


def _matrix_reference_state() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1234)
    return {
        "embed": torch.randn(_MATRIX_VOCAB, _MATRIX_EMB, generator=g),
        "linear.weight": torch.randn(_MATRIX_DENSE_OUT, _MATRIX_DENSE_IN, generator=g),
        "linear.bias": torch.randn(_MATRIX_DENSE_OUT, generator=g),
        "moe.experts": torch.randn(_MATRIX_E, _MATRIX_OUT, _MATRIX_IN, generator=g),
    }


def _matrix_write_merged(ckpt_dir: Path, ref: dict[str, torch.Tensor]) -> str:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.clone() for k, v in ref.items()}, str(ckpt_dir / "model.safetensors"))
    return str(ckpt_dir)


def _matrix_write_per_expert(ckpt_dir: Path, ref: dict[str, torch.Tensor]) -> str:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.clone() for k, v in ref.items() if k != "moe.experts"}
    for e in range(_MATRIX_E):
        state[f"moe.experts.{e}"] = ref["moe.experts"][e].clone()
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return str(ckpt_dir)


def _matrix_full_cpu(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().float().cpu()


def _matrix_local_expert_shard(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        t = t.to_local()
    return t.detach().float().cpu()


def _matrix_build_fresh(weights_path, *, loader: str, model_cls=FakeMoeModel):
    kwargs = dict(
        weights_path=weights_path,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=False,
        basic_modules=[],
    )
    if loader == "rank0":
        kwargs["broadcast_model_weights_from_rank0"] = True
    elif loader == "ep_sharded":
        kwargs["extra_parallel_stream_load"] = True
    return build_parallelize_model(model_cls(), **kwargs)


def _run_matrix_worker() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    get_torch_device().set_device(rank)
    dist.init_process_group(backend=get_dist_comm_backend())
    _init_parallel_state(dp_size=world, extra_parallel_sizes=(world,), extra_parallel_names=("ep",), dp_mode="fsdp2")

    tmp = Path(os.environ["MOE_MATRIX_TMP"])
    ref = _matrix_reference_state()
    merged_dir = str(tmp / "merged")
    per_expert_dir = str(tmp / "per_expert")
    if rank == 0:
        _matrix_write_merged(tmp / "merged", ref)
        _matrix_write_per_expert(tmp / "per_expert", ref)
    dist.barrier()

    ps = get_parallel_state()
    ep_rank = ps.extra_parallel_rank("ep")
    per = _MATRIX_E // ps.extra_parallel_sizes["ep"]
    results = []

    def check(model, tag):
        params = dict(model.named_parameters())
        got_experts = _matrix_local_expert_shard(params["moe.experts"])
        want_experts = ref["moe.experts"][ep_rank * per : (ep_rank + 1) * per]
        torch.testing.assert_close(got_experts, want_experts, atol=0.0, rtol=0.0)
        for name in ("linear.weight", "linear.bias", "embed"):
            torch.testing.assert_close(_matrix_full_cpu(params[name]), ref[name], atol=0.0, rtol=0.0)
        results.append(f"  [OK] {tag}: experts slice + dense params bit-identical to reference")

    for fmt, path, model_cls in [
        ("merged", merged_dir, FakeMoeModel),
        ("nonmerged", per_expert_dir, FakeMoeModel),
        ("nonmerged-streamable", per_expert_dir, FakeStreamableMoeModel),
    ]:
        for loader in ["plain", "rank0", "ep_sharded"]:
            tag = f"{fmt} x {loader}"
            expect_bail = fmt == "nonmerged" and loader == "ep_sharded"
            try:
                model = _matrix_build_fresh(path, loader=loader, model_cls=model_cls)
            except NotImplementedError as e:
                if expect_bail:
                    results.append(f"  [OK] {tag}: bailed as expected ({str(e)[:60]}...)")
                else:
                    results.append(f"  [FAIL] {tag}: unexpected NotImplementedError: {e}")
                dist.barrier()
                continue
            if expect_bail:
                results.append(f"  [FAIL] {tag}: expected NotImplementedError bail, but load succeeded")
                dist.barrier()
                continue
            check(model, tag)
            del model
            dist.barrier()

    if rank == 0:
        print(f"\n=== NON-PEFT MoE load matrix (world={world}, ep={world}) ===")
        print("\n".join(results))
    failed = [r for r in results if "[FAIL]" in r]
    dist.barrier()
    dist.destroy_process_group()
    if failed:
        raise SystemExit(f"{len(failed)} case(s) FAILED:\n" + "\n".join(failed))


_ACCEL = get_torch_device()


@pytest.mark.launch_plan(provider=_CI_LAUNCH_2)
@pytest.mark.skipif(
    not _ACCEL.is_available() or runner_device_count() < 2,
    reason="requires >= 2 accelerator devices (CUDA or NPU)",
)
def test_moe_extra_parallel_load_matrix(tmp_path):
    env = dict(os.environ, MOE_MATRIX_TMP=str(tmp_path))
    cmd = [
        "torchrun",
        f"--nproc_per_node={_CI_LAUNCH_2.nproc}",
        "--standalone",
        os.path.abspath(__file__),
        "--worker=matrix",
    ]
    subprocess.run(cmd, check=True, env=env)


# ======================================================================
# 2-accelerator PEFT MoE-LoRA streaming vs broadcast load
# ======================================================================


def _peft_fused_ops_override() -> str:
    from veomni.utils.import_utils import is_torch_npu_available

    backend = "fused_npu" if is_torch_npu_available() else "fused_triton"
    return f"--model.ops_implementation.moe_implementation={backend}"


def _build_and_save_peft_stream_load_assets(root_dir: str) -> None:
    """Single-process (spawned): build fused base, per-expert base, and non-zero LoRA adapters."""
    import copy
    import shutil as _shutil

    from tests.lora.test_moe_lora_trainer import _TOY_CONFIG_PATH, _yaml_for_mode
    from tests.lora.utils import build_lora_test_model
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
    from veomni.utils import helper as _helper

    _helper.set_seed(42)
    ops = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        gated_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
    )
    model = build_lora_test_model(
        config_path=_TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="cpu",
        ops_implementation=ops,
    )

    fused_dir = os.path.join(root_dir, "fused_base")
    os.makedirs(fused_dir, exist_ok=True)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(fused_dir, "model.safetensors"))
    _shutil.copy(_TOY_CONFIG_PATH, os.path.join(fused_dir, "config.json"))

    per_expert_dir = os.path.join(root_dir, "per_expert_base")
    os.makedirs(per_expert_dir, exist_ok=True)
    model.save_pretrained(per_expert_dir, safe_serialization=True)

    adapters_dir = os.path.join(root_dir, "adapters")
    for mode in ("independent", "shared"):
        with open(_yaml_for_mode(mode)) as f:
            ycfg = yaml.safe_load(f)
        lcfg = VeOmniLoraConfig.from_yaml(ycfg["model"]["lora_config"])
        peft_model = VeOmniLoraModel(copy.deepcopy(model), lcfg)
        with torch.no_grad():
            for idx, (name, p) in enumerate(peft_model.named_parameters()):
                if "lora_A" not in name and "lora_B" not in name:
                    continue
                numel = p.numel()
                vals = (
                    (torch.arange(1, numel + 1, dtype=torch.float32).remainder(997) + 1.0) / 1024.0
                    + (idx + 1) * 0.03125
                ).reshape(p.shape)
                if p.ndim == 3:
                    expert_offset = torch.arange(1, p.shape[0] + 1, dtype=torch.float32).view(-1, 1, 1) * 0.125
                    vals = vals + expert_offset
                p.copy_(vals.to(p.dtype))
                assert bool((p != 0).all().item()), f"Expected all non-zero adapter values for {name}"
        peft_model.save_pretrained(os.path.join(adapters_dir, mode), safe_serialization=True)


@pytest.fixture(scope="module")
def _peft_stream_load_assets_root(tmp_path_factory):
    import multiprocessing as mp

    from tests.lora.test_moe_lora_ep2 import _expected_qwen3_moe_toy_adapter_shapes, _load_adapter

    root_dir = str(tmp_path_factory.mktemp("stream_load_assets"))
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_build_and_save_peft_stream_load_assets, args=(root_dir,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"stream load assets build subprocess exited with code {proc.exitcode}")

    adapters_dir = os.path.join(root_dir, "adapters")
    for mode in ("independent", "shared"):
        mode_dir = os.path.join(adapters_dir, mode)
        with open(os.path.join(mode_dir, "adapter_config.json")) as f:
            acfg = json.load(f)
        assert acfg.get("veomni_lora", {}).get("moe_mode") == mode
        sd = _load_adapter(mode_dir)
        expected_shapes = _expected_qwen3_moe_toy_adapter_shapes(mode)
        assert set(sd) == set(expected_shapes)
        for k, exp_shape in expected_shapes.items():
            assert tuple(sd[k].shape) == exp_shape
            assert bool((sd[k] != 0).all().item()), f"{mode}/{k} has zero elements in fixture"

    return root_dir


def _make_adapter_resume_yaml(base_yaml: str, adapter_path: str, dest: str) -> str:
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["lora_config"]["lora_adapter"] = adapter_path
    cfg["train"]["max_steps"] = 1
    cfg["train"]["checkpoint"]["save_steps"] = 0
    cfg["train"]["checkpoint"]["save_hf_weights"] = False
    cfg["train"]["checkpoint"]["hf_save_steps"] = 0
    cfg["train"]["checkpoint"].pop("load_path", None)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f)
    return dest


_PEFT_STREAM_LOAD_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("independent", "fused"),
    ("shared", "fused"),
    ("independent", "per_expert"),
)


def _peft_scenario_argv(yaml_path: str, output_dir: str, base_dir: str, *, stream_load: bool) -> list[str]:
    argv = [
        "tests/utils/test_extra_parallel.py",
        yaml_path,
        f"--train.checkpoint.output_dir={output_dir}",
        f"--model.model_path={base_dir}",
        f"--model.config_path={base_dir}",
        _peft_fused_ops_override(),
        "--model.accelerator.ep_size=2",
    ]
    if stream_load:
        argv.extend(
            [
                "--model.broadcast_model_weights_from_rank0=False",
                "--model.extra_parallel_stream_load=True",
            ]
        )
    else:
        argv.extend(
            [
                "--model.broadcast_model_weights_from_rank0=True",
                "--model.extra_parallel_stream_load=False",
            ]
        )
    return argv


def _peft_load_and_snapshot(
    yaml_path: str, output_dir: str, base_dir: str, *, stream_load: bool
) -> dict[str, torch.Tensor]:
    from tests.lora.test_moe_lora_trainer import MoeLoraTrainer
    from veomni.arguments import VeOmniArguments, parse_args

    sys.argv = _peft_scenario_argv(yaml_path, output_dir, base_dir, stream_load=stream_load)
    args = parse_args(VeOmniArguments)
    trainer = MoeLoraTrainer(args)
    trainer.snapshot_callback._dump("pre")
    base_local = {
        n: (p.to_local() if hasattr(p, "to_local") else p).detach().cpu().clone()
        for n, p in trainer.model.named_parameters()
        if ".base_layer.weight" in n
    }
    del trainer
    gc.collect()
    if dist.is_initialized():
        dist.barrier()
    return base_local


def _run_peft_stream_load_worker(argv: list[str]) -> None:
    from tests.lora.test_moe_lora_trainer import SNAPSHOT_PRE, _compare_snapshots_bit_exact, _yaml_for_mode

    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--fused-base-dir", required=True)
    parser.add_argument("--per-expert-base-dir", required=True)
    parser.add_argument("--adapters-dir", required=True)
    parser.add_argument("--reverse-scenarios", action="store_true")
    cli_args = parser.parse_args(argv)

    scenarios = list(_PEFT_STREAM_LOAD_SCENARIOS)
    if cli_args.reverse_scenarios or os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1":
        scenarios = list(reversed(scenarios))

    local_rank = os.environ.get("LOCAL_RANK", "0")
    try:
        for mode, base_layout in scenarios:
            label = f"{mode}_{base_layout}"
            base_dir = cli_args.fused_base_dir if base_layout == "fused" else cli_args.per_expert_base_dir
            adapter_dir = os.path.join(cli_args.adapters_dir, mode)
            resume_yaml = _make_adapter_resume_yaml(
                _yaml_for_mode(mode),
                adapter_dir,
                os.path.join(cli_args.work_dir, f"resume_{label}_rank{local_rank}.yaml"),
            )
            ref_dir = os.path.join(cli_args.work_dir, f"ref_{label}")
            stream_dir = os.path.join(cli_args.work_dir, f"stream_{label}")

            ref_base_local = _peft_load_and_snapshot(resume_yaml, ref_dir, base_dir, stream_load=False)
            stream_base_local = _peft_load_and_snapshot(resume_yaml, stream_dir, base_dir, stream_load=True)

            assert set(ref_base_local) == set(stream_base_local)
            for name, ref_t in ref_base_local.items():
                assert torch.equal(stream_base_local[name], ref_t), (
                    f"[{label}] base expert local shard mismatch for {name}"
                )

            if not dist.is_initialized() or dist.get_rank() == 0:
                _compare_snapshots_bit_exact(
                    actual_path=os.path.join(stream_dir, SNAPSHOT_PRE),
                    ref_path=os.path.join(ref_dir, SNAPSHOT_PRE),
                    label=f"{mode}/{base_layout}/extra_parallel_stream_vs_broadcast",
                )
            if dist.is_initialized():
                dist.barrier()
        if dist.is_initialized():
            dist.barrier()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


@pytest.mark.launch_plan(provider=_CI_LAUNCH_2)
@pytest.mark.skipif(
    not _ACCEL.is_available() or runner_device_count() < 2,
    reason="requires >= 2 accelerator devices (CUDA or NPU)",
)
def test_peft_extra_parallel_stream_load_matches_broadcast(tmp_path, _peft_stream_load_assets_root):
    """Fused 2-rank worker comparing streaming vs broadcast PEFT MoE-LoRA load across all 3 layouts."""
    from tests.lora.test_moe_lora_trainer import _gpu_count_or_skip

    nproc = _gpu_count_or_skip(min_count=_CI_LAUNCH_2.nproc, max_count=_CI_LAUNCH_2.nproc)
    work_dir = str(tmp_path / "stream_load_work")
    os.makedirs(work_dir, exist_ok=True)
    fused_base_dir = os.path.join(_peft_stream_load_assets_root, "fused_base")
    per_expert_base_dir = os.path.join(_peft_stream_load_assets_root, "per_expert_base")
    adapters_dir = os.path.join(_peft_stream_load_assets_root, "adapters")

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--standalone",
        os.path.abspath(__file__),
        "--worker=peft",
        f"--work-dir={work_dir}",
        f"--fused-base-dir={fused_base_dir}",
        f"--per-expert-base-dir={per_expert_base_dir}",
        f"--adapters-dir={adapters_dir}",
    ]
    res = subprocess.run(cmd, check=True)
    assert res.returncode == 0


# ======================================================================
# ExtraParallel gradient-norm clipping (4/8-accelerator + CPU unit tests)
# ======================================================================

_CI_LAUNCH_4 = LaunchSpec(nproc=4)
_CI_LAUNCH_8 = LaunchSpec(nproc=8)

logger = helper.create_logger(__name__)


def _torch_npu_version() -> str | None:
    """Return the version of torch_npu if installed, otherwise return None."""
    if importlib.util.find_spec("torch_npu") is None:
        return None
    return str(importlib.metadata.version("torch_npu"))


class ToyMoeAndEmbedModel(torch.nn.Module):
    """Toy model with MoE+Embedding where every param is 1 and forward returns their sum.

    Each param's local backward gradient is always 1 (no all2all in MoE forward, so EP
    param grad accumulation only occurs across ep_fsdp ranks).
    """

    _no_split_modules = ["ToyMoeAndEmbedDecoderLayer", "ToyEmbed"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = ToyEmbed()
        self.bias = torch.nn.Parameter(torch.ones(16), requires_grad=True)
        self.decoder = ToyMoeAndEmbedDecoderLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x + self.bias).sum()
        loss = loss + self.embed_tokens() + self.decoder()
        return loss

    def init_weights(self):
        self.embed_tokens.weight.data.fill_(1.0)
        self.decoder.embed_tokens.weight.data.fill_(1.0)
        self.bias.data.fill_(1.0)
        self.decoder.regular_mlp.data.fill_(1.0)
        self.decoder.moe.experts.data.fill_(1.0)

    def get_parallel_plan(self):
        ep_plan = {"decoder.moe.experts": Shard(0)}
        emb_plan = {"embed_tokens.weight": Shard(0), "decoder.embed_tokens.weight": Shard(0)}
        parallel_plan = ParallelPlan(
            extra_parallel_plan={
                "ep": ep_plan,
                "emb": emb_plan,
            }
        )
        parallel_plan.extra_parallel_fsdp_no_shard_module = {
            "ep": {"decoder.moe"},
            "emb": {"embed_tokens", "decoder.embed_tokens"},
        }
        return parallel_plan


class ToyEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.weight.sum()


class ToyMoeAndEmbedDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = ToyEmbed()
        self.regular_mlp = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)
        self.moe = ToyMoeExperts()

    def forward(self) -> torch.Tensor:
        return self.embed_tokens() + self.regular_mlp.sum() + self.moe()


class ToyMoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Parameter(torch.ones(64, 16, 32), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.experts.sum()


# (label, ep_size, emb_size, dp_replicate_size, cpu_offload[, extra_parallel_only])
_FSDP2_4GPU_SCENARIOS: tuple[
    tuple[str, int, int, int | None, bool] | tuple[str, int, int, int | None, bool, bool], ...
] = (
    ("fsdp2_no_extra_no_offload", 1, 1, None, False),
    ("fsdp2_no_extra_cpu_offload", 1, 1, None, True),
    ("fsdp2_ep2_no_offload", 2, 1, None, False),
    ("fsdp2_ep2_cpu_offload", 2, 1, None, True),
    ("fsdp2_ep4_no_offload", 4, 1, None, False),
    ("fsdp2_ep4_cpu_offload", 4, 1, None, True),
    ("fsdp2_emb4_no_offload", 1, 4, None, False),
    ("fsdp2_emb4_cpu_offload", 1, 4, None, True),
    ("fsdp2_ep2_emb2_no_offload", 2, 2, None, False),
    ("fsdp2_ep2_emb2_cpu_offload", 2, 2, None, True),
    ("fsdp2_ep2_emb4_no_offload", 2, 4, None, False),
    ("fsdp2_ep2_emb4_cpu_offload", 2, 4, None, True),
    ("extra_parallel_only_emb4", 1, 4, None, False, True),
)

_HSDP_8GPU_SCENARIOS: tuple[tuple[str, int, int, int | None, bool], ...] = (
    ("hsdp_no_extra_no_offload", 1, 1, 2, False),
    ("hsdp_no_extra_cpu_offload", 1, 1, 2, True),
    ("hsdp_ep4_no_offload", 4, 1, 2, False),
    ("hsdp_ep4_cpu_offload", 4, 1, 2, True),
    ("hsdp_emb4_no_offload", 1, 4, 2, False),
    ("hsdp_emb4_cpu_offload", 1, 4, 2, True),
    ("hsdp_ep2_emb4_no_offload", 2, 4, 2, False),
    ("hsdp_ep2_emb4_cpu_offload", 2, 4, 2, True),
)


def _hsdp_extra_parallel_buggy_on_npu() -> bool:
    npu_ver = _torch_npu_version()
    return bool(npu_ver and Version(npu_ver) < Version("2.9.0"))


def _run_clip_scenario(
    *,
    label: str,
    world_size: int,
    ep_size: int,
    emb_size: int,
    dp_replicate_size: int | None,
    cpu_offload: bool,
    extra_parallel_only: bool = False,
) -> None:
    dp_replicate = dp_replicate_size or 1
    assert world_size % dp_replicate == 0
    dp_shard = world_size // dp_replicate

    state_name = f"clip_w{world_size}_rep{dp_replicate}_ep{ep_size}_emb{emb_size}_epo{int(extra_parallel_only)}"
    ps = _init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=dp_replicate,
        dp_shard_size=dp_shard,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        extra_parallel_sizes=(ep_size, emb_size),
        extra_parallel_placement_innermost=(False, False),
        extra_parallel_names=("ep", "emb"),
        dp_mode="fsdp2",
        extra_parallel_only=extra_parallel_only,
        name=state_name,
    )

    with use_parallel_state(ps):
        assert get_parallel_state() is ps
        model_args = ModelArguments(config_path="test")
        torch.manual_seed(42)
        model = ToyMoeAndEmbedModel()
        model = build_parallelize_model(
            model,
            extra_parallel_only=extra_parallel_only,
            init_device="meta",
            weights_path=None,
            mixed_precision=(
                MixedPrecisionConfig(enable=False)
                if extra_parallel_only
                else model_args.accelerator.fsdp_config.mixed_precision
            ),
            enable_gradient_checkpointing=False,
            enable_fsdp_offload=cpu_offload,
            basic_modules=[],
            enable_reentrant=False,
            enable_forward_prefetch=model_args.accelerator.fsdp_config.forward_prefetch,
            broadcast_model_weights_from_rank0=False,
            max_load_broadcast_size=model_args.accelerator.fsdp_config.max_load_broadcast_size,
        )

        fsdp_group = ps.fsdp_group
        ep_group = ps.extra_parallel_group("ep") if ps.extra_parallel_enabled("ep") else None
        emb_group = ps.extra_parallel_group("emb") if ps.extra_parallel_enabled("emb") else None

        ep_fsdp_group = None
        if ps.extra_parallel_group("ep") and ps.extra_parallel_fsdp_device_mesh["ep"] is not None:
            ep_fsdp_group = ps.extra_parallel_fsdp_device_mesh["ep"]["ep_fsdp"].get_group()
            assert dist.get_world_size(group=ep_fsdp_group) == dp_shard // ep_size, (
                f"[{label}] expected ep_fsdp group size {dp_shard // ep_size}, got {dist.get_world_size(group=ep_fsdp_group)}"
            )

        emb_fsdp_group = None
        if ps.extra_parallel_group("emb") and ps.extra_parallel_fsdp_device_mesh["emb"] is not None:
            emb_fsdp_group = ps.extra_parallel_fsdp_device_mesh["emb"]["emb_fsdp"].get_group()
            assert dist.get_world_size(group=emb_fsdp_group) == dp_shard // emb_size, (
                f"[{label}] expected emb_fsdp group size {dp_shard // emb_size}, got {dist.get_world_size(group=emb_fsdp_group)}"
            )
            if emb_size == 2 and dp_shard == 4:
                assert dist.get_world_size(group=emb_fsdp_group) == 2

        optimizer = build_optimizer(
            model,
            lr=model_args.optimizer.lr,
            weight_decay=model_args.optimizer.weight_decay,
            fused=True,
            optimizer_type=model_args.optimizer.type,
            no_decay_modules=model_args.optimizer.no_decay_modules,
            no_decay_params=model_args.optimizer.no_decay_params,
        )
        logger.info_rank0(
            "[%s] group sizes - fsdp: %s, ep: %s, ep_fsdp: %s, emb: %s, emb_fsdp: %s",
            label,
            dist.get_world_size(group=fsdp_group) if fsdp_group is not None else None,
            dist.get_world_size(group=ep_group) if ep_group is not None else None,
            dist.get_world_size(group=ep_fsdp_group) if ep_fsdp_group is not None else None,
            dist.get_world_size(group=emb_group) if emb_group is not None else None,
            dist.get_world_size(group=emb_fsdp_group) if emb_fsdp_group is not None else None,
        )
        device_type = get_device_type()
        tensor_device = torch.device(f"{device_type}:{get_device_id()}")
        max_grad_norm = model_args.optimizer.max_grad_norm

        def check_model_param_grad_one_by_one(target_model, expected_grad, ep_expected_grad, emb_expected_grad, msg):
            for name, param in target_model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
                if name == "decoder.moe.experts":
                    torch.testing.assert_close(
                        grad_local,
                        torch.full_like(grad_local, ep_expected_grad),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=f"[{label}] ({msg}) Gradient mismatch for {name}, shape {grad_local.shape}, expected {ep_expected_grad}",
                    )
                elif "embed_tokens" in name:
                    torch.testing.assert_close(
                        grad_local,
                        torch.full_like(grad_local, emb_expected_grad),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=f"[{label}] ({msg}) Gradient mismatch for {name}, shape {grad_local.shape}, expected {emb_expected_grad}",
                    )
                else:
                    torch.testing.assert_close(
                        grad_local,
                        torch.full_like(grad_local, expected_grad),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=f"[{label}] ({msg}) Gradient mismatch for {name}, shape {grad_local.shape}, expected {expected_grad}",
                    )

        for step in range(2):
            inputs = torch.ones(1, 16, device=tensor_device)
            loss = model(inputs)
            loss.backward()

            expected = 1.0
            ep_expected = 1.0 / ps.extra_parallel_sizes["ep"]
            emb_expected = 1.0 / ps.extra_parallel_sizes["emb"]
            check_model_param_grad_one_by_one(
                model,
                expected_grad=expected,
                ep_expected_grad=ep_expected,
                emb_expected_grad=emb_expected,
                msg=f"Before clipping step={step}",
            )
            if extra_parallel_only:
                # Replicated params are not reduced by FSDP: assign distinct per-rank
                # grads whose mean is still 1 so post-clip checks pass only if averaged.
                model.bias.grad.mul_(2.0 * dist.get_rank() / (world_size - 1))

            expected_total_grad_norm = math.sqrt(
                (64 * 16 + 64 * 16) * (1 / ps.extra_parallel_sizes["emb"] ** 2)
                + 16
                + 64 * 16
                + (64 * 16 * 32) * (1 / ps.extra_parallel_sizes["ep"] ** 2)
            )
            total_grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)

            torch.testing.assert_close(
                total_grad_norm_pre_clip.item(),
                expected=expected_total_grad_norm,
                atol=1e-6,
                rtol=1e-6,
                msg=f"[{label}] step={step} total grad norm mismatch",
            )

            clip_coeff = min(max_grad_norm / expected_total_grad_norm, 1.0)
            ep_clip_coeff = 1.0 / ps.extra_parallel_sizes["ep"] * clip_coeff
            emb_clip_coeff = 1.0 / ps.extra_parallel_sizes["emb"] * clip_coeff
            check_model_param_grad_one_by_one(
                model,
                expected_grad=clip_coeff,
                ep_expected_grad=ep_clip_coeff,
                emb_expected_grad=emb_clip_coeff,
                msg=f"After clipping step={step}",
            )
            model.zero_grad()

        del optimizer, model
        gc.collect()
        dist.barrier()


def _run_clip_grad_norm_worker(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["fsdp2_4gpu", "hsdp_8gpu"], required=True)
    parser.add_argument("--reverse-scenarios", action="store_true")
    cli_args = parser.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    get_torch_device().set_device(f"{get_device_type()}:{local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    world_size = dist.get_world_size()

    if cli_args.suite == "fsdp2_4gpu":
        scenarios = list(_FSDP2_4GPU_SCENARIOS)
    else:
        scenarios = [
            s for s in _HSDP_8GPU_SCENARIOS if not (_hsdp_extra_parallel_buggy_on_npu() and (s[1] > 1 or s[2] > 1))
        ]

    reverse = cli_args.reverse_scenarios or os.environ.get("VEOMNI_REVERSE_SCENARIOS") == "1"
    if reverse:
        scenarios = list(reversed(scenarios))

    try:
        for scenario in scenarios:
            label, ep_size, emb_size, dp_replicate_size, cpu_offload, *extra = scenario
            _run_clip_scenario(
                label=label,
                world_size=world_size,
                ep_size=ep_size,
                emb_size=emb_size,
                dp_replicate_size=dp_replicate_size,
                cpu_offload=cpu_offload,
                extra_parallel_only=bool(extra[0]) if extra else False,
            )
        dist.barrier()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            clear_parallel_state()


def _run_clip_grad_norm_suite(suite: str, nproc: int, reverse_scenarios: bool = False) -> None:
    command = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--standalone",
        os.path.abspath(__file__),
        "--worker=clip_grad_norm",
        f"--suite={suite}",
    ]
    if reverse_scenarios:
        command.append("--reverse-scenarios")
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


@pytest.mark.launch_plan(provider=_CI_LAUNCH_4)
@pytest.mark.skipif(
    not _ACCEL.is_available() or runner_device_count() < 4,
    reason="requires >= 4 accelerator devices (CUDA or NPU)",
)
def test_clip_grad_norm_fsdp2_4gpu():
    """Fused 4-rank FSDP2 worker covering no-extra, EP2, EP4, emb4, and EP2+emb4 across no_offload/cpu_offload."""
    _run_clip_grad_norm_suite("fsdp2_4gpu", nproc=_CI_LAUNCH_4.nproc)


@pytest.mark.launch_plan(provider=_CI_LAUNCH_8)
@pytest.mark.skipif(
    not _ACCEL.is_available() or runner_device_count() < 8,
    reason="requires >= 8 accelerator devices (CUDA or NPU)",
)
def test_clip_grad_norm_hsdp_8gpu():
    """Fused 8-rank HSDP worker covering (2, 4) no-extra, EP4, emb4, and EP2+emb4 across no_offload/cpu_offload."""
    _run_clip_grad_norm_suite("hsdp_8gpu", nproc=_CI_LAUNCH_8.nproc)


# Direct ``importlib`` lookup: ``veomni.distributed.fsdp2.__init__`` shadows the submodule
# name with ``from .clip_grad_norm import clip_grad_norm``.
cgn_mod = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
extra_parallel_fsdp2_clip_grad_norm = cgn_mod.extra_parallel_fsdp2_clip_grad_norm


@dataclass
class _FakeParallelState:
    """Minimal stub for ``get_parallel_state()`` consumed by clip_grad_norm."""

    extra_parallel_names: list[str] = field(default_factory=lambda: ["ep"])
    fsdp_group: object | None = None
    extra_parallel_fsdp_device_mesh: dict[str, object | None] = field(default_factory=dict)
    dp_mode: str = "none"

    def extra_parallel_enabled(self, name: str) -> bool:
        return False

    def extra_parallel_group(self, name: str) -> object | None:
        return None

    def extra_parallel_flat_group(self, name: str) -> object | None:
        return None


def _make_model(
    param_groups: dict[str, list[nn.Parameter]],
    replicated_ids: set,
    persistent_ids: set | None = None,
) -> nn.Module:
    m = nn.Module()
    m._extra_parallel_param_groups = param_groups
    m._ep_replicated_lora_param_ids = replicated_ids
    m._persistent_extra_parallel_param_ids = persistent_ids or set()
    return m


def _param_with_grad(grad: torch.Tensor) -> nn.Parameter:
    p = nn.Parameter(torch.zeros_like(grad), requires_grad=True)
    p.grad = grad.clone()
    return p


@pytest.mark.parametrize("extra_names", [["ep"], ["ep", "emb"]])
def test_inf_norm_with_replicated_bucket_does_not_crash(extra_names):
    """``norm_type=inf`` with both EP and EP-replicated buckets populated."""
    p_non = _param_with_grad(torch.tensor([1.0, -2.0, 0.5]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=list(extra_names),
        extra_parallel_fsdp_device_mesh=dict.fromkeys(extra_names),
    )
    grads_seen = [p_non.grad.abs().max()]

    param_groups: dict[str, list[nn.Parameter]] = {"non_extra_parallel": [p_non]}
    replicated_ids: set = set()
    for i, name in enumerate(extra_names):
        p_ep = _param_with_grad(torch.tensor([3.0 + i, -1.0, 0.0]))
        p_ep_rep = _param_with_grad(torch.tensor([-(7.0 + i), 0.1, 4.0]))
        param_groups[name] = [p_ep, p_ep_rep]
        replicated_ids.add(id(p_ep_rep))
        grads_seen.append(p_ep.grad.abs().max())
        grads_seen.append(p_ep_rep.grad.abs().max())

    model = _make_model(param_groups, replicated_ids)
    expected = torch.stack(grads_seen).max().to(torch.float32)
    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(expected.item(), abs=1e-6), (
        f"inf-norm mismatch with {extra_names=}: got {total.item()}, expected {expected.item()}"
    )


def test_inf_norm_single_axis_no_replicated_still_works():
    """Sanity: the legacy K1+K2==1 happy path still passes."""
    p_non = _param_with_grad(torch.tensor([0.25]))
    p_ep = _param_with_grad(torch.tensor([5.0, -6.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ep"],
        extra_parallel_fsdp_device_mesh={"ep": None},
    )
    model = _make_model({"non_extra_parallel": [p_non], "ep": [p_ep]}, replicated_ids=set())

    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(6.0, abs=1e-6)


def test_persistent_extra_parallel_norm_reduces_once_over_flat_mesh():
    """A persistent 2D shard owns unique values and needs one flat-mesh reduction."""
    p_non = _param_with_grad(torch.tensor([12.0]))
    p_persistent = _param_with_grad(torch.tensor([3.0, 4.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ple"],
        extra_parallel_fsdp_device_mesh={"ple": None},
    )
    fake_ps.extra_parallel_enabled = lambda _name: True
    fake_ps.extra_parallel_group = lambda name: f"{name}_group"
    fake_ps.extra_parallel_flat_group = lambda name: f"{name}_flat_group"
    model = _make_model(
        {"non_extra_parallel": [p_non], "ple": [p_persistent]},
        replicated_ids=set(),
        persistent_ids={id(p_persistent)},
    )
    reduce_calls = []

    def record_reduce(params, norm_type, reduce_groups):
        reduce_calls.append((list(params), list(reduce_groups)))
        return cgn_mod._local_pth_sum(params, float(norm_type))

    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
        mock.patch.object(cgn_mod, "_fsdp2_reduce_group", side_effect=record_reduce),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=2.0, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(13.0, abs=1e-6)
    persistent_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_persistent]
    assert len(persistent_calls) == 1
    assert persistent_calls[0][1] == [("ple_flat", "ple_flat_group")]


def test_persistent_ple_and_ep_norm_use_independent_reduction_groups():
    """PLE uses one flat reduction while ordinary EP uses EP-FSDP then EP."""

    class _FakeMesh:
        def __init__(self, group):
            self.group = group

        def get_group(self):
            return self.group

    p_non = _param_with_grad(torch.tensor([12.0]))
    p_ple = _param_with_grad(torch.tensor([3.0, 4.0]))
    p_ep = _param_with_grad(torch.tensor([5.0, 12.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ple", "ep"],
        extra_parallel_fsdp_device_mesh={
            "ple": {"ple_fsdp": _FakeMesh("ple_fsdp_group")},
            "ep": {"ep_fsdp": _FakeMesh("ep_fsdp_group")},
        },
    )
    fake_ps.extra_parallel_enabled = lambda _name: True
    fake_ps.extra_parallel_group = lambda name: f"{name}_group"
    fake_ps.extra_parallel_flat_group = lambda name: f"{name}_flat_group"
    model = _make_model(
        {
            "non_extra_parallel": [p_non],
            "ple": [p_ple],
            "ep": [p_ep],
        },
        replicated_ids=set(),
        persistent_ids={id(p_ple)},
    )
    reduce_calls = []

    def record_reduce(params, norm_type, reduce_groups):
        reduce_calls.append((list(params), list(reduce_groups)))
        return cgn_mod._local_pth_sum(params, float(norm_type))

    with (
        mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(cgn_mod, "get_device_type", return_value="cpu"),
        mock.patch.object(cgn_mod, "_fsdp2_reduce_group", side_effect=record_reduce),
    ):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=2.0, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(math.sqrt(338.0), abs=1e-6)
    ple_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_ple]
    ep_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_ep]
    assert len(ple_calls) == 1
    assert ple_calls[0][1] == [("ple_flat", "ple_flat_group")]
    assert len(ep_calls) == 1
    assert ep_calls[0][1] == [("ep_fsdp", "ep_fsdp_group"), ("ep", "ep_group")]


def test_omni_module_clip_grad_norm_extra_parallel_only_averages_replicated_grads_and_skips_fsdp_sum():
    """``veomni_omni_module_clip_grad_norm`` averages replicated grads on ``fsdp_group``
    and skips the ``fsdp`` norm reduction on ``non_extra_parallel`` when
    ``ParallelState.extra_parallel_only`` is set."""
    omni_cgn_mod = importlib.import_module("veomni.distributed.clip_grad_norm")

    class _FakeMesh:
        def __init__(self, group):
            self.group = group

        def get_group(self):
            return self.group

    p_non = _param_with_grad(torch.tensor([3.0, 4.0]))
    p_emb = _param_with_grad(torch.tensor([5.0, 12.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["emb"],
        fsdp_group="fsdp_group",
        extra_parallel_fsdp_device_mesh={"emb": {"emb_fsdp": _FakeMesh("emb_fsdp_group")}},
        dp_mode="fsdp2",
    )
    fake_ps.any_extra_parallel_enabled = True
    fake_ps.sp_size = 1
    fake_ps.extra_parallel_only = True
    fake_ps.extra_parallel_enabled = lambda _name: True
    fake_ps.extra_parallel_group = lambda name: f"{name}_group"

    model = _make_model(
        {"non_extra_parallel": [p_non], "emb": [p_emb]},
        replicated_ids=set(),
    )

    reduce_calls = []
    avg_calls = []

    def record_reduce(params, norm_type, reduce_groups):
        reduce_calls.append((list(params), list(reduce_groups)))
        return cgn_mod._local_pth_sum(params, float(norm_type))

    def record_avg(m):
        avg_calls.append(m)

    with (
        mock.patch.object(omni_cgn_mod, "get_parallel_state", return_value=fake_ps),
        mock.patch.object(omni_cgn_mod, "_fsdp2_reduce_group", side_effect=record_reduce),
        mock.patch.object(omni_cgn_mod, "_average_extra_parallel_only_replicated_grads", side_effect=record_avg),
    ):
        total = omni_cgn_mod.veomni_omni_module_clip_grad_norm(model, max_norm=1e9, norm_type=2.0)

    assert avg_calls == [model]
    assert total == pytest.approx(math.sqrt(25.0 + 169.0), abs=1e-6)
    non_ep_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_non]
    emb_calls = [call for call in reduce_calls if len(call[0]) == 1 and call[0][0] is p_emb]
    assert len(non_ep_calls) == 1
    assert non_ep_calls[0][1] == []
    assert len(emb_calls) == 1
    assert emb_calls[0][1] == [("emb_fsdp", "emb_fsdp_group"), ("emb", "emb_group")]


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("matrix", "peft", "clip_grad_norm"), default="matrix")
    mode_args, rest = parser.parse_known_args()
    if mode_args.worker == "matrix":
        os.environ.setdefault("MOE_MATRIX_TMP", "/tmp/moe_ep_matrix_fixture")
        _run_matrix_worker()
        print("ALL CASES PASSED", file=sys.stderr)
    elif mode_args.worker == "peft":
        _run_peft_stream_load_worker(rest)
    else:
        _run_clip_grad_norm_worker(rest)
