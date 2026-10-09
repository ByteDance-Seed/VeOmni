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

"""Per-expert -> fused stacking under ``load_model_weights_ep_sharded``, on CPU.

Each EP rank is simulated in turn with a stub parallel state over a model whose
expert parameters already hold the rank-local ``[E/ep, ...]`` shape, mirroring
Qwen3.5-MoE: the trunk experts are fused in the checkpoint while the MTP experts
are stored per expert and go through ``Qwen3MoeCheckpointTensorConverter``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import Shard

import veomni.models.module_utils as module_utils
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter


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

    def __init__(self) -> None:
        super().__init__()
        num_local = NUM_EXPERTS // EP_SIZE
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_layer(num_local)])
        self.mtp = nn.Module()
        self.mtp.layers = nn.ModuleList([_layer(num_local)])
        self.lm_head = nn.Linear(HIDDEN, 5, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(
            extra_parallel_plan={
                "ep": {
                    f"{root}.layers.*.mlp.experts.{proj}": Shard(0)
                    for root in ("model", "mtp")
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
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: parallel_state)

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

    monkeypatch.setattr(module_utils, "safe_open", _RecordingSafeOpen)
    module_utils.load_model_weights_ep_sharded(model, str(weights_path), init_device="cpu")
    return read_keys


def _meta_model(converter_cls=Qwen3MoeCheckpointTensorConverter) -> nn.Module:
    with torch.device("meta"):
        model = _MtpMoeModel()
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


def test_converter_expert_range_stacks_only_its_experts():
    converter = Qwen3MoeCheckpointTensorConverter(num_experts=4)
    name = "model.layers.0.mlp.experts.3.up_proj.weight"
    assert converter.fused_expert_target(name) == ("model.layers.0.mlp.experts.gate_up_proj", 3)
    assert converter.fused_expert_target("model.layers.0.mlp.experts.gate_up_proj") is None

    local = converter.for_expert_range(2, 2)
    tensors = {expert: torch.full((1, 2), float(expert)) for expert in (2, 3)}
    assert local.convert("model.layers.0.mlp.experts.2.down_proj.weight", tensors[2]) is None
    converted = local.convert("model.layers.0.mlp.experts.3.down_proj.weight", tensors[3])
    assert converted.name == "model.layers.0.mlp.experts.down_proj"
    torch.testing.assert_close(converted.tensor, torch.stack([tensors[2], tensors[3]]))
    assert local.finalize() == []

    with pytest.raises(ValueError, match="outside this converter's range"):
        local.convert("model.layers.0.mlp.experts.1.down_proj.weight", tensors[2])
    with pytest.raises(ValueError, match="outside this converter's 4 experts"):
        converter.for_expert_range(3, 2)
