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
# See the License for the specific language governing limitations
# under the License.

"""CPU tests for modeling-side MoE fallbacks."""

from __future__ import annotations

import torch


def test_merged_experts_act_fn_ep_keeps_empty_tokens_on_graph():
    from veomni.models.utils.moe_utils import _MergedExpertsActFnEP

    hidden = 4
    permute_tokens = torch.zeros(0, hidden, dtype=torch.float32, requires_grad=True)
    cumsum = torch.zeros(2, dtype=torch.long)
    gate_up_proj = torch.randn(2, hidden * 2, hidden, requires_grad=True)
    down_proj = torch.randn(2, hidden, hidden, requires_grad=True)
    output = _MergedExpertsActFnEP.apply(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        torch.nn.functional.silu,
    )
    assert output.shape == (0, hidden)
    assert output.requires_grad
    output.sum().backward()
    assert permute_tokens.grad is not None
    assert permute_tokens.grad.shape == permute_tokens.shape
