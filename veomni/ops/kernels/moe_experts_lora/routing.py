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

"""Routing helpers shared by eager MoE-LoRA implementations."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor


def group_routing_assignments(
    selected_experts: Tensor,
    num_experts: int,
) -> Iterator[tuple[int, Tensor, Tensor]]:
    """Yield non-empty expert groups as ``(expert, top-k slot, token)``."""
    num_tokens = selected_experts.shape[0]
    flat_experts = selected_experts.T.reshape(-1)
    sorted_experts, flat_positions = torch.sort(flat_experts, stable=True)
    expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)

    offset = 0
    for expert_idx, count in zip(expert_ids.tolist(), counts.tolist(), strict=True):
        group_positions = flat_positions[offset : offset + count]
        offset += count
        if expert_idx == num_experts:
            continue
        yield expert_idx, group_positions // num_tokens, group_positions % num_tokens
