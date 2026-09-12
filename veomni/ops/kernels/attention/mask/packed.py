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

"""Shared packed-sequence predicates for SDPA and FlexAttention masks."""

from collections.abc import Callable

import torch
from torch import Tensor

from ..helper import require_all


def packed_mask_function(
    *,
    mask_function: Callable,
    q_length: int,
    kv_length: int,
    q_offset: int,
    kv_offset: int,
    cu_seqlens: Tensor,
    cu_seqlens_k: Tensor | None,
    device: torch.device | str,
) -> Callable:
    """Compose ``mask_function`` with independent packed Q/K metadata.

    HF passes offset-adjusted indices to mask functions. Packed cached attention
    can have different query and key segment lengths, so one shared segment-id
    tensor cannot classify both sides. Map every query to its position in the
    corresponding key segment before evaluating the causal/sliding predicate;
    queries are the suffix of each key segment, matching cached-attention
    ``q_offset=kv_length-q_length`` semantics.
    """
    if not isinstance(cu_seqlens, Tensor):
        raise TypeError(f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens).__name__}")
    if cu_seqlens_k is not None and not isinstance(cu_seqlens_k, Tensor):
        raise TypeError(f"cu_seqlens_k must be a torch.Tensor, got {type(cu_seqlens_k).__name__}")
    if cu_seqlens_k is None and q_length != kv_length:
        raise ValueError("packed SDPA/FlexAttention with q_length != kv_length requires cu_seqlens_k")
    cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if cu_seqlens.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "packed SDPA/FlexAttention requires the same number of query and key segments, got "
            f"{cu_seqlens.numel() - 1} and {cu_seqlens_k.numel() - 1}"
        )

    device = torch.device(device)
    cu_seqlens_q = _validated_cu_seqlens(cu_seqlens, q_length, device)
    cu_seqlens_k = _validated_cu_seqlens(cu_seqlens_k, kv_length, device)
    q_segment_ids = _segment_ids(cu_seqlens_q, q_length, device)
    k_segment_ids = _segment_ids(cu_seqlens_k, kv_length, device)

    q_positions = torch.arange(q_length, device=device)
    q_starts = cu_seqlens_q[:-1]
    k_starts = cu_seqlens_k[:-1]
    q_lengths = cu_seqlens_q[1:] - q_starts
    k_lengths = cu_seqlens_k[1:] - k_starts
    q_positions_in_k = (
        k_starts[q_segment_ids]
        + k_lengths[q_segment_ids]
        - q_lengths[q_segment_ids]
        + q_positions
        - q_starts[q_segment_ids]
    )

    def packed_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """Require paired segments and evaluate the base mask in key coordinates."""
        q_local = q_idx - q_offset
        k_local = kv_idx - kv_offset
        same_segment = q_segment_ids[q_local] == k_segment_ids[k_local]
        mapped_q_idx = q_positions_in_k[q_local] + kv_offset
        return same_segment & mask_function(batch_idx, head_idx, mapped_q_idx, kv_idx)

    return packed_mask


def _validated_cu_seqlens(cu_seqlens: Tensor, length: int, device: torch.device) -> Tensor:
    """Require integer, strictly increasing, full-coverage cumulative lengths."""
    if not isinstance(cu_seqlens, Tensor):
        raise TypeError(f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens).__name__}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must have shape [n_seg + 1], got {tuple(cu_seqlens.shape)}")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cu_seqlens must have dtype int32 or int64, got {cu_seqlens.dtype}")
    cu_seqlens = cu_seqlens.to(device=device)
    require_all(
        (cu_seqlens[:1] == 0) & (cu_seqlens[-1:] == length),
        f"cu_seqlens must run from 0 to {length}",
    )
    require_all(cu_seqlens[1:] > cu_seqlens[:-1], "cu_seqlens must be strictly increasing")
    return cu_seqlens


def _segment_ids(cu_seqlens: Tensor, length: int, device: torch.device) -> Tensor:
    """Expand validated cumulative lengths into one segment id per token."""
    return torch.bucketize(torch.arange(length, device=device), cu_seqlens[1:], right=True)
