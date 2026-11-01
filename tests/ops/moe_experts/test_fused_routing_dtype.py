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
