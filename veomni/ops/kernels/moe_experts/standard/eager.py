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
# See the License for the specific language governing limitations
# under the License.

"""standard MoE experts eager math (fused operator order)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_weight: Tensor,
    fc1_2_weight: Tensor,
    fc2_weight: Tensor,
    fc1_1_2_weight: Tensor,
    *,
    num_experts: int,
    swiglu_limit: float | None = None,
) -> Tensor:
    """Routed expert MLP. Empty ``fc1_*`` means that layout is unused.

    Routing weights scale the SwiGLU intermediate, then ``fc2``. Regular
    autograd, no custom backward.
    """
    has_split = fc1_1_weight.numel() > 0
    has_merged = fc1_1_2_weight.numel() > 0
    if has_split == has_merged:
        raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both or neither.")
    if has_split and fc1_2_weight.numel() == 0:
        raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")

    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        if has_merged:
            gate, up = F.linear(x, fc1_1_2_weight[idx]).chunk(2, dim=-1)
        else:
            gate = F.linear(x, fc1_1_weight[idx])
            up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.silu(gate) * up
        # Fused order: scale the intermediate, then fc2.
        y = y * routing_weights[token_idx, top_k_pos, None]
        y = F.linear(y, fc2_weight[idx])
        output = output.index_add(0, token_idx, y.to(output.dtype))
    return output
