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

"""Batch-local top-k index helpers shared by DSA fused adapters."""

from __future__ import annotations

import torch


def local_topk_to_global(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    """Shift per-batch top-k indices into a flattened KV sequence.

    Indices outside ``[0, seqlen_k)`` stay invalid as ``-1``. An out-of-range
    local index must not wrap into the next batch's keys.
    """
    if topk_indices.dim() != 3:
        raise ValueError(f"topk_indices must be [B, S_q, topk], got {tuple(topk_indices.shape)}")
    batch_offsets = torch.arange(topk_indices.shape[0], device=topk_indices.device, dtype=torch.int32).view(-1, 1, 1)
    batch_offsets = batch_offsets * int(seqlen_k)
    topk_i32 = topk_indices.to(torch.int32)
    valid = (topk_i32 >= 0) & (topk_i32 < int(seqlen_k))
    return torch.where(valid, topk_i32 + batch_offsets, topk_i32.new_full((), -1))
