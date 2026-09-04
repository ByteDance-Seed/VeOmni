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

"""gpt_oss MoE experts eager math (HF ``GptOssExperts`` loop)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    gate_up_proj: Tensor,
    gate_up_proj_bias: Tensor,
    down_proj: Tensor,
    down_proj_bias: Tensor,
    *,
    num_experts: int,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> Tensor:
    """Routed GPT-OSS expert MLP. Routing weights are applied after ``down_proj``.

    Weights are right-multiplied (``x @ W + b``), matching HF ``GptOssExperts``.
    Regular autograd, no custom backward.
    """
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        current = hidden_states[token_idx]
        gate_up = current @ gate_up_proj[idx] + gate_up_proj_bias[idx]
        gate = gate_up[..., ::2].clamp(max=limit)
        up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
        gated = (up + 1) * (gate * torch.sigmoid(gate * alpha))
        out = gated @ down_proj[idx] + down_proj_bias[idx]
        # HF GptOssExperts applies routing after the down projection.
        weighted = out * routing_weights[token_idx, top_k_pos, None]
        output = output.index_add(0, token_idx, weighted.to(output.dtype))
    return output
