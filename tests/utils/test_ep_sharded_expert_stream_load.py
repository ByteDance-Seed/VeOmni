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
expert parameters already hold the rank-local ``[E/ep, ...]`` shape. Most cases
mirror Qwen3.5-MoE: the trunk experts are fused in the checkpoint while the MTP
experts are stored per expert and go through ``Qwen3MoeCheckpointTensorConverter``.
The all-per-expert trunk case covers the Qwen3-MoE, DeepSeek-V3 and
Qwen3-Omni-MoE checkpoint layouts.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import Shard

import veomni.models.load_plan as load_plan
import veomni.models.module_utils as module_utils
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.models.transformers.deepseek_v3.checkpoint_tensor_converter import DeepseekV3CheckpointTensorConverter
from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import DeepseekV4CheckpointTensorConverter
from veomni.models.transformers.glm_moe_dsa.checkpoint_tensor_converter import GlmMoeDsaCheckpointTensorConverter
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter
from veomni.models.transformers.qwen3_omni_moe.checkpoint_tensor_converter import (
    Qwen3OmniMoeCheckpointTensorConverter,
)


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


def _load_as_rank(
    monkeypatch,
    weights_path: Path,
    ep_rank: int,
    model: nn.Module,
    sliced_reads: Optional[dict] = None,
    read_keys: Optional[list] = None,
) -> list:
    """Stream ``weights_path`` into ``model`` as ``ep_rank``; return the keys read whole.

    ``sliced_reads``, when given, collects ``{key: index}`` for every partial read
    (a ``get_slice`` handle that was indexed, not just asked for its shape).
    ``read_keys``, when given, is the list the whole reads are appended to, so a
    caller can inspect it after the load raises.
    """
    parallel_state = SimpleNamespace(
        extra_parallel_enabled=lambda name: True,
        extra_parallel_sizes={"ep": EP_SIZE},
        extra_parallel_rank=lambda name: ep_rank,
    )
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: parallel_state)

    if read_keys is None:
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

                def get_slice(self, key):
                    sl = handle.get_slice(key)

                    class _Slice:
                        def __getitem__(self, index):
                            if sliced_reads is not None:
                                sliced_reads[key] = index
                            return sl[index]

                        def __getattr__(self, attr):
                            return getattr(sl, attr)

                    return _Slice()

                def __getattr__(self, attr):
                    return getattr(handle, attr)

            outer._proxy = _Proxy()
            return outer._proxy

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    monkeypatch.setattr(module_utils, "safe_open", _RecordingSafeOpen)
    monkeypatch.setattr(load_plan, "safe_open", _RecordingSafeOpen)
    module_utils.load_model_weights_ep_sharded(model, str(weights_path), init_device="cpu")
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


GATE_UP = "model.layers.0.mlp.experts.gate_up_proj"
DOWN = "model.layers.0.mlp.experts.down_proj"


class _FusedMoeModel(nn.Module):
    """Fused trunk experts, a dense head and a persistent buffer; no converter."""

    _create_checkpoint_tensor_converter = None

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_layer(NUM_EXPERTS // EP_SIZE)])
        self.model.register_buffer("rope_scale", torch.empty(HIDDEN))
        self.lm_head = nn.Linear(HIDDEN, 5, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(
            extra_parallel_plan={
                "ep": {f"model.layers.*.mlp.experts.{p}": Shard(0) for p in ("gate_up_proj", "down_proj")}
            }
        )


def _write_fused_checkpoint(path: Path, num_experts: int = NUM_EXPERTS) -> dict:
    generator = torch.Generator().manual_seed(0)
    state = {
        GATE_UP: torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, generator=generator),
        DOWN: torch.randn(num_experts, HIDDEN, INTERMEDIATE, generator=generator),
        "lm_head.weight": torch.randn(5, HIDDEN, generator=generator),
        "model.rope_scale": torch.randn(HIDDEN, generator=generator),
        # The model builds no MTP layer.
        "mtp.layers.0.mlp.experts.down_proj": torch.randn(num_experts, HIDDEN, INTERMEDIATE, generator=generator),
    }
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path / "model.safetensors"))
    return state


def _spy_load_plan(monkeypatch) -> list:
    plans = []
    execute = module_utils.execute_load_plan

    def _execute(plan, *args, **kwargs):
        plans.append(plan)
        return execute(plan, *args, **kwargs)

    monkeypatch.setattr(module_utils, "execute_load_plan", _execute)
    return plans


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_a_converter_free_checkpoint_streams_through_the_load_plan(monkeypatch, tmp_path, ep_rank):
    state = _write_fused_checkpoint(tmp_path)
    with torch.device("meta"):
        model = _FusedMoeModel()
    plans = _spy_load_plan(monkeypatch)
    sliced = {}

    read_keys = _load_as_rank(monkeypatch, tmp_path, ep_rank, model, sliced)

    assert len(plans) == 1
    num_local = NUM_EXPERTS // EP_SIZE
    rows = slice(ep_rank * num_local, (ep_rank + 1) * num_local)
    experts = model.model.layers[0].mlp.experts
    torch.testing.assert_close(experts.gate_up_proj.data, state[GATE_UP][rows], atol=0, rtol=0)
    torch.testing.assert_close(experts.down_proj.data, state[DOWN][rows], atol=0, rtol=0)
    torch.testing.assert_close(model.lm_head.weight.data, state["lm_head.weight"], atol=0, rtol=0)
    torch.testing.assert_close(model.model.rope_scale, state["model.rope_scale"], atol=0, rtol=0)
    # Experts are read as this rank's rows only; the unbuilt MTP tensor is never read.
    assert sorted(read_keys) == ["lm_head.weight", "model.rope_scale"]
    assert {key: index[0] for key, index in sliced.items()} == {GATE_UP: rows, DOWN: rows}


def test_a_converter_that_claims_no_key_streams_through_the_load_plan(monkeypatch, tmp_path):
    """Qwen3-MoE's converter fuses per-expert keys; a fused checkpoint gives it nothing to do."""
    state = _write_fused_checkpoint(tmp_path)
    with torch.device("meta"):
        model = _FusedMoeModel()
    model._create_checkpoint_tensor_converter = lambda m: Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
    plans = _spy_load_plan(monkeypatch)

    _load_as_rank(monkeypatch, tmp_path, 1, model)

    assert len(plans) == 1
    torch.testing.assert_close(
        model.model.layers[0].mlp.experts.down_proj.data, state[DOWN][NUM_EXPERTS // EP_SIZE :], atol=0, rtol=0
    )


def test_a_converter_claimed_key_keeps_the_converter_loop(monkeypatch, tmp_path):
    _write_checkpoint(tmp_path)
    plans = _spy_load_plan(monkeypatch)

    _load_as_rank(monkeypatch, tmp_path, 0, _meta_model())

    assert plans == []


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_a_load_plan_shape_mismatch_raises_on_every_rank_before_any_read(monkeypatch, tmp_path, ep_rank):
    _write_fused_checkpoint(tmp_path, num_experts=NUM_EXPERTS + EP_SIZE)
    with torch.device("meta"):
        model = _FusedMoeModel()
    sliced, read_keys = {}, []

    with pytest.raises(RuntimeError, match=r"has shape \(6, \d, \d\), but model tensor .* is \(4, \d, \d\) in total"):
        _load_as_rank(monkeypatch, tmp_path, ep_rank, model, sliced, read_keys)

    assert sliced == {} and read_keys == []
    assert all(p.is_meta for p in model.parameters())


def test_a_load_plan_rejects_two_keys_loading_one_tensor(tmp_path):
    save_file({"a.weight": torch.zeros(2), "b.weight": torch.zeros(2)}, str(tmp_path / "model.safetensors"))

    with pytest.raises(RuntimeError, match="both load model tensor 'w'"):
        load_plan.build_load_plan(
            {"a.weight": "model.safetensors", "b.weight": "model.safetensors"},
            {"model.safetensors": str(tmp_path / "model.safetensors")},
            target_name=lambda key: "w",
            parameter_layouts={"w": load_plan.LoadLayout.whole((2,))},
            buffer_shapes={},
        )


def test_load_layout_dim0_shard_covers_its_rows_of_the_whole_tensor():
    layout = load_plan.LoadLayout.dim0_shard((2, 4, 3), rank=1, world_size=3)

    assert layout.global_shape == (6, 4, 3)
    assert layout.region == (slice(2, 4), slice(0, 4), slice(0, 3))
    assert not layout.is_whole
    assert load_plan.LoadLayout.dim0_shard((2, 4), rank=0, world_size=1).is_whole
    assert load_plan.LoadLayout.whole(()).is_whole


class _ClaimsByDtype:
    def can_handle(self, name):
        return False

    def can_handle_tensor(self, name, tensor):
        return tensor.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    "spec_info, converter, expected",
    [
        pytest.param({}, None, True, id="no_converter"),
        pytest.param({}, Qwen3MoeCheckpointTensorConverter(NUM_EXPERTS), True, id="converter_claims_nothing"),
        pytest.param({}, Qwen3MoeCheckpointTensorConverter(NUM_EXPERTS), False, id="converter_claims_a_key"),
        # Whether it claims a key depends on tensor dtypes the plan cannot see by name.
        pytest.param({}, _ClaimsByDtype(), False, id="converter_claims_by_dtype"),
        # Persistent ExtraParallel parameters (Qwen4-Exp PLE) carry their own DTensor layout.
        pytest.param(
            {"model.ple.weight": SimpleNamespace(persistent_fsdp_shard_dim=1)}, None, False, id="persistent_param"
        ),
    ],
)
def test_which_checkpoints_the_load_plan_streams(request, spec_info, converter, expected):
    model = SimpleNamespace(_fqn2spec_info=spec_info)
    names = [GATE_UP, "lm_head.weight"]
    if request.node.callspec.id == "converter_claims_a_key":
        names.append(PER_EXPERT.format(expert=0, proj="gate_proj"))

    assert module_utils._load_plan_streams_checkpoint(model, converter, names) is expected
