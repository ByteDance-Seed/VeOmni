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

"""independent MoE-LoRA eager math. Regular autograd, no custom backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..routing import group_routing_assignments


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    *,
    num_experts: int,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """Routed SwiGLU with a per-expert LoRA pair per logical spec."""
    scale_down = hidden_states.new_tensor(lora_scale_down)

    output = torch.zeros_like(hidden_states)
    for expert_idx, top_k_pos, token_idx in group_routing_assignments(selected_experts, num_experts):
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, fc1_1_2_weight[expert_idx]).chunk(2, dim=-1)
        gate_hidden = F.linear(current_state, lora_a_gate[expert_idx])
        up_hidden = F.linear(current_state, lora_a_up[expert_idx])
        gate = torch.addmm(gate, gate_hidden, lora_b_gate[expert_idx].T, alpha=lora_scale_gate)
        up = torch.addmm(up, up_hidden, lora_b_up[expert_idx].T, alpha=lora_scale_up)
        mid = F.silu(gate) * up
        mid = mid * routing_weights[token_idx, top_k_pos, None]
        lora_x_down = F.linear(F.linear(mid, lora_a_down[expert_idx]), lora_b_down[expert_idx]) * scale_down
        current_hidden_states = F.linear(mid, fc2_weight[expert_idx]) + lora_x_down
        output.index_add_(0, token_idx, current_hidden_states.to(output.dtype))
    return output
