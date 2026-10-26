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

"""Fused MoE entries cast fp32 router scores onto the activation dtype."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    routing = torch.ones(2, 1, dtype=torch.float32)
    selected = torch.zeros(2, 1, dtype=torch.int64)
    fc1 = torch.randn(2, 8, 4)
    fc2 = torch.randn(2, 4, 4)
    return hidden, routing, selected, fc1, fc2


def test_triton_casts_routing_weights(monkeypatch):
    from veomni.ops.kernels.moe_experts.standard import triton

    hidden, routing, selected, fc1, fc2 = _inputs()
    seen: dict[str, torch.dtype] = {}

    def apply(*args):
        seen["dtype"] = args[1].dtype
        return hidden

    monkeypatch.setattr(triton.TritonFusedMoeExpertFunction, "apply", apply)
    triton.group_gemm_fused_moe_forward(2, routing, selected, hidden, fc1, fc1, fc2)
    assert seen["dtype"] == torch.bfloat16


def test_quack_casts_routing_weights(monkeypatch):
    from veomni.ops.kernels.moe_experts.standard import quack

    hidden, routing, selected, fc1, fc2 = _inputs()
    seen: dict[str, torch.dtype] = {}

    def apply(*args):
        seen["dtype"] = args[1].dtype
        return hidden

    monkeypatch.setattr(quack.QuackFusedMoeExpertFunction, "apply", apply)
    quack.quack_gemm_fused_moe_forward(2, routing, selected, hidden, fc1, fc1, fc2)
    assert seen["dtype"] == torch.bfloat16


def test_gpt_oss_quack_casts_routing_weights(monkeypatch):
    from veomni.ops.kernels.moe_experts.gpt_oss import quack

    hidden, routing, selected, fc1, fc2 = _inputs()
    bias = torch.randn(2, 4, dtype=torch.bfloat16)
    seen: dict[str, torch.dtype] = {}

    def apply(*args):
        seen["dtype"] = args[1].dtype
        return hidden

    monkeypatch.setattr(quack.GptOssQuackFusedMoeExpertFunction, "apply", apply)
    quack.quack_gemm_gpt_oss_fused_moe_forward(2, routing, selected, hidden, fc1, bias, fc2, bias)
    assert seen["dtype"] == torch.bfloat16


def test_npu_casts_routing_weights(monkeypatch):
    from veomni.ops.kernels.moe_experts.standard import npu

    hidden, routing, selected, fc1, fc2 = _inputs()
    seen: dict[str, torch.dtype] = {}

    def forward(*args, **kwargs):
        seen["dtype"] = args[1].dtype
        return hidden

    monkeypatch.setattr(npu, "_npu_fused_moe_forward", forward)
    npu.npu_fused_moe_forward(2, routing, selected, hidden, fc1, fc1, fc2)
    assert seen["dtype"] == torch.bfloat16


def test_mlu_casts_routing_weights(monkeypatch):
    from veomni.ops.kernels.moe_experts.standard import mlu

    hidden, routing, selected, fc1, fc2 = _inputs()
    seen: dict[str, torch.dtype] = {}
    permute_mod = types.ModuleType("apex.contrib.permute")
    unpermute_mod = types.ModuleType("apex.contrib.unpermute")
    permute_mod.permute = lambda hidden_states, experts, dim: (hidden_states, torch.arange(hidden_states.shape[0]))

    def unpermute(tokens, indices, routing_weights):
        seen["dtype"] = routing_weights.dtype
        return tokens

    unpermute_mod.unpermute = unpermute
    monkeypatch.setitem(sys.modules, "apex", types.ModuleType("apex"))
    monkeypatch.setitem(sys.modules, "apex.contrib", types.ModuleType("apex.contrib"))
    monkeypatch.setitem(sys.modules, "apex.contrib.permute", permute_mod)
    monkeypatch.setitem(sys.modules, "apex.contrib.unpermute", unpermute_mod)

    def apply(*args, **kwargs):
        return hidden

    monkeypatch.setattr(mlu.MLUGroupGemm, "apply", apply)
    mlu.mlu_group_gemm_fused_moe_forward(2, routing, selected, hidden, fc1, fc1, fc2)
    assert seen["dtype"] == torch.bfloat16


@pytest.mark.parametrize("variant", ["shared", "independent"])
def test_lora_triton_casts_routing_weights(monkeypatch, variant: str):
    from veomni.ops.kernels.moe_experts_lora.independent import triton as indep_triton
    from veomni.ops.kernels.moe_experts_lora.shared import triton as shared_triton

    hidden, routing, selected, fc1, fc2 = _inputs()
    seen: dict[str, torch.dtype] = {}

    def apply(*args):
        seen["dtype"] = args[1].dtype
        return hidden

    if variant == "shared":
        monkeypatch.setattr(shared_triton.MergedFc1TritonFusedLoRAMoeExpertFunction, "apply", apply)
        loras = [torch.randn(2, 4, dtype=torch.bfloat16) for _ in range(6)]
        shared_triton.group_gemm_fused_lora_moe_forward(2, routing, selected, hidden, fc1, fc2, *loras, 0.5, 0.5, 0.5)
    else:
        monkeypatch.setattr(indep_triton.MergedFc1IndependentTritonFusedLoRAMoeExpertFunction, "apply", apply)
        loras = [torch.randn(2, 2, 4, dtype=torch.bfloat16) for _ in range(6)]
        indep_triton.group_gemm_fused_independent_lora_moe_forward(
            2, routing, selected, hidden, fc1, fc2, *loras, 0.5, 0.5, 0.5
        )
    assert seen["dtype"] == torch.bfloat16


def test_lora_npu_helper_casts_routing_weights(monkeypatch):
    from veomni.ops.kernels.moe_experts_lora import helper

    hidden, routing, selected, fc1, fc2 = _inputs()
    seen: list[torch.dtype] = []

    npu_mod = types.ModuleType("torch_npu")
    npu_mod.npu_moe_token_permute = lambda h, s: (h, torch.arange(h.shape[0]))
    npu_mod.npu_swiglu = lambda x, dim=-1: x[:, : x.shape[-1] // 2]

    def unpermute(output, row_ids_map, probs=None):
        seen.append(probs.dtype)
        return output

    npu_mod.npu_moe_token_unpermute = unpermute
    monkeypatch.setitem(sys.modules, "torch_npu", npu_mod)

    from veomni.ops.kernels.moe_experts.standard import npu as std_npu

    monkeypatch.setattr(std_npu, "npu_group_gemm", lambda a, b, g: a)
    monkeypatch.setattr(
        std_npu,
        "dispatch_preprocess",
        lambda s, e, g: ([1], [1], torch.ones(1, e, dtype=torch.int64), torch.ones(e, dtype=torch.int64)),
    )
    monkeypatch.setattr(std_npu, "alltoall_dispatch", lambda h, *args: (h, torch.arange(h.shape[0])))

    def combine(output, routing_weights, *args):
        seen.append(routing_weights.dtype)
        return output

    monkeypatch.setattr(std_npu, "alltoall_combine", combine)

    def zero_delta(x, _g):
        return torch.zeros_like(x)

    helper.npu_non_ep_forward(2, routing, selected, hidden, fc1, fc2, zero_delta, zero_delta)
    helper.npu_ep_forward(2, routing, selected, hidden, fc1, fc2, zero_delta, zero_delta)
    assert seen == [torch.bfloat16, torch.bfloat16]


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_lora_moe_layers_preserve_routing_weight_dtype(mode: str):
    from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts

    class _BaseExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.hidden_dim = 4
            self.intermediate_dim = 4
            self.act_fn = torch.nn.SiLU()
            self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 4, dtype=torch.bfloat16))
            self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4, dtype=torch.bfloat16))

    wrapper_cls = LoraSharedExperts if mode == "shared" else LoraIndependentExperts
    layer = wrapper_cls(_BaseExperts(), r=2, lora_alpha=4)
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    top_k_index = torch.zeros(3, 1, dtype=torch.long)
    top_k_weights = torch.ones(3, 1, dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    class _RecordingOp:
        impl = "fused_triton"

        def __call__(self, *args, **kwargs):
            captured["routing_weights"] = args[1]
            return torch.zeros_like(hidden)

    layer.veomni_moe_lora = _RecordingOp()
    layer(hidden, top_k_index, top_k_weights)
    assert captured["routing_weights"] is top_k_weights
    assert captured["routing_weights"].dtype == torch.float32


def test_unpermute_preserves_tokens_dtype_with_fp32_routing_weights():
    from veomni.distributed.moe.moe_utils import unpermute

    tokens = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    routing_weights = torch.full((2, 2), 0.5, dtype=torch.float32, requires_grad=True)
    routing_map = torch.ones((2, 2), dtype=torch.long)
    permutation_mapping = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    out = unpermute(tokens, routing_weights, torch.Size([2, 8]), permutation_mapping, routing_map)
    assert out.dtype == torch.bfloat16
    out.sum().backward()
    assert tokens.grad is not None and tokens.grad.dtype == torch.bfloat16
    assert routing_weights.grad is not None and routing_weights.grad.dtype == torch.float32
