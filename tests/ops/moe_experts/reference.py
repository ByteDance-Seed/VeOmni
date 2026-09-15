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

"""Independent PyTorch references shared by MoE and expert-parallel tests."""

import torch
import torch.nn.functional as F
from torch import Tensor


def standard_fused_reference(
    hidden: Tensor,
    routing: Tensor,
    selected: Tensor,
    fc1_1: Tensor,
    fc1_2: Tensor,
    fc2: Tensor,
    *,
    num_experts: int,
    swiglu_limit: float | None = None,
) -> Tensor:
    """Evaluate standard MoE with the routing-before-fc2 fused-kernel order.

    Triton, Quack, and expert-parallel grouped GEMMs use this order. It is
    algebraically equivalent to routing after a bias-free fc2, although BF16
    rounding can differ.
    """
    output = torch.zeros_like(hidden)
    expert_mask = F.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden[token_idx]
        gate = F.linear(x, fc1_1[idx])
        up = F.linear(x, fc1_2[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        intermediate = F.silu(gate) * up
        intermediate = intermediate * routing[token_idx, top_k_pos, None]
        expert_output = F.linear(intermediate, fc2[idx])
        output.index_add_(0, token_idx, expert_output.to(output.dtype))
    return output
