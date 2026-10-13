# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""``ep_sharded_stream_load`` for the bf16 DeepSeek-V4.1 llm.

The split checkpoint ships packed fp4 experts and fp8 dense weights, each with
its ``<param>_scale``. Streaming must spend those scales the way the
whole-tensor loader does: a rank reads only its experts and their scale rows,
and every dense weight with its whole scale. The model here is a stand-in with
the llm's key names and plan, so the test needs neither the modeling nor a GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed.tensor import Shard

import veomni.models.module_utils as module_utils
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.models.module_utils import load_model_weights_ep_sharded
from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import (
    DeepseekV41DequantConverter,
    dequant_converted_weight,
)
from veomni.models.seed_omni.modules.deepseek_v41.dequant import dequant_fp8_blockwise


N_EXPERTS = 4
HIDDEN = 64
INTER = 32
EXPERTS = "model.layers.0.mlp.experts.gate_up_proj"
DENSE = "model.layers.0.self_attn.wq_a.weight"
NORM = "model.norm.weight"


def _released_checkpoint(path: str, *, with_dense_scale: bool = True) -> dict:
    torch.manual_seed(0)
    state = {
        EXPERTS: torch.randint(-128, 128, (N_EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.int8),
        f"{EXPERTS}_scale": torch.rand(N_EXPERTS, 2 * INTER, HIDDEN // 32) + 0.5,
        DENSE: torch.randn(HIDDEN, HIDDEN).to(torch.float8_e4m3fn),
        f"{DENSE}_scale": torch.rand(HIDDEN // 32, HIDDEN // 32) + 0.5,
        NORM: torch.randn(HIDDEN),
    }
    if not with_dense_scale:
        del state[f"{DENSE}_scale"]
    save_file(state, f"{path}/model.safetensors")
    return state


def _holder(**params: torch.Tensor) -> nn.Module:
    module = nn.Module()
    for name, tensor in params.items():
        module.register_parameter(name, nn.Parameter(tensor))
    return module


class _Llm(nn.Module):
    """The llm's names, shapes and EP plan, holding ``experts_per_rank`` experts."""

    def __init__(self, experts_per_rank: int) -> None:
        super().__init__()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _holder(
            gate_up_proj=torch.empty(experts_per_rank, 2 * INTER, HIDDEN, dtype=torch.bfloat16)
        )
        layer.self_attn = nn.Module()
        layer.self_attn.wq_a = _holder(weight=torch.empty(HIDDEN, HIDDEN, dtype=torch.bfloat16))
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.model.norm = _holder(weight=torch.empty(HIDDEN))
        self.config = SimpleNamespace(num_hidden_layers=40)

    def get_parallel_plan(self):
        return ParallelPlan(extra_parallel_plan={"ep": {"model.layers.*.mlp.experts.gate_up_proj": Shard(0)}})

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return DeepseekV41DequantConverter(model.config)


def _load(tmp_path, monkeypatch, *, ep_size: int, ep_rank: int) -> dict:
    """Load as EP rank ``ep_rank`` of ``ep_size`` would, in one process."""
    monkeypatch.setattr(
        module_utils, "_ep_dim0_slice_meta", lambda _state, _group, target0: (ep_size, ep_rank, target0 * ep_size)
    )
    model = _Llm(N_EXPERTS // ep_size)
    load_model_weights_ep_sharded(model, str(tmp_path), "cpu")
    return {name: param.data for name, param in model.named_parameters()}


class TestStreamingSpendsTheScales:
    def test_each_rank_dequantizes_only_its_experts(self, tmp_path, monkeypatch):
        state = _released_checkpoint(str(tmp_path))
        want = dequant_converted_weight(state[EXPERTS], state[f"{EXPERTS}_scale"])
        shards = [_load(tmp_path, monkeypatch, ep_size=2, ep_rank=rank)[EXPERTS] for rank in range(2)]
        torch.testing.assert_close(torch.cat(shards), want, rtol=0, atol=0)

    def test_a_dense_weight_is_dequantized_whole(self, tmp_path, monkeypatch):
        state = _released_checkpoint(str(tmp_path))
        loaded = _load(tmp_path, monkeypatch, ep_size=1, ep_rank=0)
        want = dequant_fp8_blockwise(state[DENSE], state[f"{DENSE}_scale"])
        torch.testing.assert_close(loaded[DENSE], want, rtol=0, atol=0)

    def test_an_unquantized_tensor_loads_as_is(self, tmp_path, monkeypatch):
        state = _released_checkpoint(str(tmp_path))
        loaded = _load(tmp_path, monkeypatch, ep_size=1, ep_rank=0)
        torch.testing.assert_close(loaded[NORM], state[NORM], rtol=0, atol=0)

    def test_a_quantized_weight_without_its_scale_is_refused(self, tmp_path, monkeypatch):
        """Read raw, fp8 would cast into the bf16 parameter unscaled."""
        _released_checkpoint(str(tmp_path), with_dense_scale=False)
        with pytest.raises(NotImplementedError, match="whole-tensor"):
            _load(tmp_path, monkeypatch, ep_size=1, ep_rank=0)


class TestConverterDeclaresTheCompanions:
    def test_a_weight_streams_with_its_scale(self):
        converter = DeepseekV41DequantConverter()
        assert converter.dim0_rowwise_companion(EXPERTS) == f"{EXPERTS}_scale"
        assert converter.dim0_rowwise_companion(f"{EXPERTS}_scale") is None
        assert converter.should_skip_without_loading(f"{EXPERTS}_scale")
        assert not converter.should_skip_without_loading(EXPERTS)

    def test_a_shrunk_stack_offers_no_companions(self):
        """Its keys are renamed, which only the whole-tensor loader applies."""
        converter = DeepseekV41DequantConverter()
        converter._layer_map, converter._engram_map = {0: 0, 20: 1}, {}
        assert converter.dim0_rowwise_companion(EXPERTS) is None
        assert not converter.should_skip_without_loading(f"{EXPERTS}_scale")
