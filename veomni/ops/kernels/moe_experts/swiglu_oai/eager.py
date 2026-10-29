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

"""SwiGLU-OAI MoE experts eager math (Hugging Face MiniMax M3 operator order)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_merged_layout(fc1_1_weight: Tensor, fc1_2_weight: Tensor, fc1_1_2_weight: Tensor) -> None:
    if fc1_1_weight.numel() or fc1_2_weight.numel() or not fc1_1_2_weight.numel():
        raise ValueError("SwiGLU-OAI MoE experts require the merged fc1_1_2_weight and no split fc1 weights.")


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
    swiglu_alpha: float,
    swiglu_limit: float | None = None,
) -> Tensor:
    """Routed SwiGLU-OAI expert MLP on merged fc1_1_2_weight.

    Matches the Hugging Face MiniMax M3 expert loop: clamp, the SwiGLU-OAI
    activation, fc2, then the routing weights in their own dtype. Fused rows
    scale before fc2 (Triton) or after it (NPU), so they agree with this row
    only to rounding. Regular autograd, no custom backward.
    """
    _check_merged_layout(fc1_1_weight, fc1_2_weight, fc1_1_2_weight)
    output = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        gate, up = F.linear(hidden_states[token_idx], fc1_1_2_weight[idx]).chunk(2, dim=-1)
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        glu = gate * torch.sigmoid(gate * swiglu_alpha)
        y = F.linear((up + 1.0) * glu, fc2_weight[idx]) * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, y.to(output.dtype))
    return output
