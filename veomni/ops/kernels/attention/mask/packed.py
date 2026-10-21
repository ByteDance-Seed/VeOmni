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

"""Shared packed-sequence predicates for eager and FlexAttention masks."""

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
    batch_size: int = 1,
) -> Callable:
    """Compose ``mask_function`` with independent packed Q/K metadata.

    HF passes offset-adjusted indices to mask functions. Packed cached attention
    can have different query and key segment lengths, so one shared segment-id
    tensor cannot classify both sides. Map every query to its position in the
    corresponding key segment before evaluating the causal/sliding predicate;
    queries are the suffix of each key segment, matching cached-attention
    ``q_offset=kv_length-q_length`` semantics. Self-attention that shares one
    ``cu_seqlens`` object at equal Q/K lengths validates and expands it once.

    ``q_length`` and ``kv_length`` are one row. ``cu_seqlens`` may use those
    per-row coordinates, or the batch-flattened coordinates whose last entry is
    ``batch_size * length`` (a ``(2, 8)`` batch is ``[0, 8, 16]``). Flattened
    coordinates are looked up with ``batch_idx`` and must cut on every row
    boundary, since one dense mask row cannot represent a segment that crosses
    into the next batch row.
    """
    if not isinstance(cu_seqlens, Tensor):
        raise TypeError(f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens).__name__}")
    if cu_seqlens_k is not None and not isinstance(cu_seqlens_k, Tensor):
        raise TypeError(f"cu_seqlens_k must be a torch.Tensor, got {type(cu_seqlens_k).__name__}")
    if cu_seqlens_k is None and q_length != kv_length:
        raise ValueError("packed eager/FlexAttention with q_length != kv_length requires cu_seqlens_k")
    cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if cu_seqlens.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "packed eager/FlexAttention requires the same number of query and key segments, got "
            f"{cu_seqlens.numel() - 1} and {cu_seqlens_k.numel() - 1}"
        )

    device = torch.device(device)
    share_qk = cu_seqlens is cu_seqlens_k and q_length == kv_length
    cu_seqlens_q, q_rows = _validated_cu_seqlens(cu_seqlens, q_length, device, batch_size)
    if share_qk:
        cu_seqlens_k, k_rows = cu_seqlens_q, q_rows
    else:
        cu_seqlens_k, k_rows = _validated_cu_seqlens(cu_seqlens_k, kv_length, device, batch_size)
    if q_rows != k_rows:
        raise ValueError(
            "packed eager/FlexAttention requires query and key cumulative lengths to cover the same "
            f"number of batch rows, got {q_rows} and {k_rows}"
        )
    rows = q_rows
    q_segment_ids = _segment_ids(cu_seqlens_q, q_length * rows, device)
    k_segment_ids = q_segment_ids if share_qk else _segment_ids(cu_seqlens_k, kv_length * rows, device)

    q_positions = torch.arange(q_length * rows, device=device)
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
        if rows == 1:
            q_flat, k_flat = q_local, k_local
            mapped_q_idx = q_positions_in_k[q_flat] + kv_offset
        else:
            q_flat = batch_idx * q_length + q_local
            k_flat = batch_idx * kv_length + k_local
            mapped_q_idx = q_positions_in_k[q_flat] - batch_idx * kv_length + kv_offset
        same_segment = q_segment_ids[q_flat] == k_segment_ids[k_flat]
        return same_segment & mask_function(batch_idx, head_idx, mapped_q_idx, kv_idx)

    return packed_mask


def _validated_cu_seqlens(
    cu_seqlens: Tensor, length: int, device: torch.device, batch_size: int = 1
) -> tuple[Tensor, int]:
    """Require integer, strictly increasing cumulative lengths covering one row or the batch."""
    if not isinstance(cu_seqlens, Tensor):
        raise TypeError(f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens).__name__}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must have shape [n_seg + 1], got {tuple(cu_seqlens.shape)}")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cu_seqlens must have dtype int32 or int64, got {cu_seqlens.dtype}")
    cu_seqlens = cu_seqlens.to(device=device)
    require_all(cu_seqlens[1:] > cu_seqlens[:-1], "cu_seqlens must be strictly increasing")
    total = int(cu_seqlens[-1].item())
    if int(cu_seqlens[0].item()) != 0 or total not in (length, length * batch_size):
        expected = f"0 to {length}" if batch_size == 1 else f"0 to {length} or {length * batch_size}"
        raise ValueError(f"cu_seqlens must run from {expected}")
    rows = 1 if total == length else batch_size
    if rows > 1:
        boundaries = torch.arange(1, batch_size, device=cu_seqlens.device, dtype=cu_seqlens.dtype) * length
        present = (cu_seqlens.unsqueeze(0) == boundaries.unsqueeze(1)).any(dim=1)
        if not bool(present.all()):
            missing = boundaries[~present].tolist()
            raise ValueError(
                "batch-flattened cu_seqlens must include every row boundary "
                f"({length}, ..., {length * (batch_size - 1)}), missing {missing}"
            )
    return cu_seqlens, rows


def _segment_ids(cu_seqlens: Tensor, length: int, device: torch.device) -> Tensor:
    """Expand validated cumulative lengths into one segment id per token."""
    return torch.bucketize(torch.arange(length, device=device), cu_seqlens[1:], right=True)
