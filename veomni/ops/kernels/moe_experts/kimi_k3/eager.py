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

"""Kimi K3 situ MoE experts eager implementation (HF expert loop)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    *,
    num_experts: int,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> Tensor:
    """Routed SituGLU expert loop over the merged ``gate_up`` layout.

    Regular autograd, no custom backward. The SituGLU activation computes in
    fp32 exactly like the modeling-side ``SituAndMul`` reference.
    """
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate, up = F.linear(x, fc1_1_2_weight[idx]).chunk(2, dim=-1)
        gate = gate.to(torch.float32)
        up = up.to(torch.float32)
        situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        y = F.linear((situ_a * up).to(hidden_states.dtype), fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output = output.index_add(0, token_idx, y.to(output.dtype))
    return output
