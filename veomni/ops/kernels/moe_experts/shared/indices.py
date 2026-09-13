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

"""Routing index helpers shared by Quack standard and gpt_oss."""

from __future__ import annotations

import torch

from .scatter import compute_expert_scatter_index


def build_moe_indices(expert_index: torch.Tensor, num_experts: int):
    """Build cu_seqlens_m, A_idx, and scatter_index from expert routing.

    Args:
        expert_index: [T, topk] expert assignments.
        num_experts: total number of experts.

    Returns:
        cu_seqlens_m: [E+1] cumulative token counts per expert (int32).
        A_idx: [T*topk] token indices sorted by expert assignment (int32).
        scatter_index: [T, topk] indices for moe_gather/moe_scatter (int32).
    """
    topk = expert_index.shape[1]
    sorted_order, scatter_index = compute_expert_scatter_index(expert_index)
    A_idx = (sorted_order // topk).int()

    splits = torch.bincount(expert_index.reshape(-1), minlength=num_experts)
    cu_seqlens_m = torch.zeros(num_experts + 1, dtype=torch.int32, device=expert_index.device)
    cu_seqlens_m[1:] = torch.cumsum(splits, dim=0).int()

    return cu_seqlens_m, A_idx, scatter_index


def cumsum_to_cu_seqlens(cumsum: torch.Tensor) -> torch.Tensor:
    """Convert [E] cumsum to [E+1] cu_seqlens_m with leading zero (int32)."""
    zero = torch.zeros(1, dtype=torch.int32, device=cumsum.device)
    return torch.cat([zero, cumsum.int()])
