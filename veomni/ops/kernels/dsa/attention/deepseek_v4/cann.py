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

"""DeepSeek-V4 sparse MQA on Ascend CANN (``cann_ops_transformer`` sparse FlashMLA).

The kernel encodes the contract's visibility as mask modes instead of
candidate lists. ``ori_mask_mode=4`` with ``ori_win_left=sliding_window - 1``
is the causal sliding window over the full-resolution rows.
``cmp_mask_mode=3`` lets segment row ``t`` see that segment's compressed rows
``[0, (t + 1) // compress_ratio)``. HCA layers stop there, and CSA layers also
narrow to the indexer's selection.

Both modes count from each segment's first row, so ``position_ids`` must
restart at 0 in every packed segment and equal the row index otherwise. The
``cann`` indexer checks this on CSA layers. Packed (TND) selections address
each segment's own compressed rows while the contract addresses the packed
buffer, so this adapter undoes the indexer's rebase.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import accumulate

import torch
from torch import Tensor

from ...sparse_indices import CompressedCandidates
from ...vendor.npu_sparse_flash_mla import npu_sparse_flash_mla
from . import eager


_ORI_SLIDING_CAUSAL_MASK_MODE = 4
_CMP_SEGMENT_CAUSAL_MASK_MODE = 3


@lru_cache(maxsize=8)
def _segment_compressed_starts(
    slices: tuple[tuple[int, int], ...], compress_ratio: int, device: torch.device
) -> Tensor:
    """Return each query row's first packed compressed row, ``[1, S, 1]`` int32."""
    lengths = [end - start for start, end in slices]
    starts = [0, *accumulate(length // compress_ratio for length in lengths)][:-1]
    rows = torch.repeat_interleave(
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.long),
        output_size=slices[-1][1],
    )
    return rows.to(device).view(1, -1, 1)


def wrapper(
    q: Tensor,
    ori_kv: Tensor,
    cmp_kv: Tensor | None,
    attn_sink: Tensor,
    position_ids: Tensor,
    *,
    sliding_window: int,
    compress_ratio: int,
    candidates: CompressedCandidates | None = None,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    query_offset: int = 0,
    attention_mask: Tensor | None = None,
    sm_scale: float | None = None,
    return_lse: bool = False,
    dropout: float = 0.0,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    """Run CANN sparse FlashMLA over the window and compressed selection.

    Layers without compressed rows run the eager row. ``attention_mask`` is
    not read, because the mask modes already carry causality and packed
    isolation, so padded batches are unsupported. The kernel needs one row per
    query (no KV cache or context-parallel offset) and validates its own
    dtype, window, ratio and head-size limits. Packed calls take the TND path,
    which needs batch size 1 and reads the slice boundaries on the host.
    """
    if cmp_kv is None or cmp_kv.shape[1] == 0:
        return eager.wrapper(
            q,
            ori_kv,
            cmp_kv,
            attn_sink,
            position_ids,
            sliding_window=sliding_window,
            compress_ratio=compress_ratio,
            candidates=candidates,
            packed_sequence_slices=packed_sequence_slices,
            query_offset=query_offset,
            attention_mask=attention_mask,
            sm_scale=sm_scale,
            return_lse=return_lse,
            dropout=dropout,
            return_attn_weights=return_attn_weights,
        )
    del position_ids, attention_mask
    if dropout:
        raise ValueError("cann DeepSeek-V4 sparse attention requires dropout=0.")
    if return_attn_weights:
        raise ValueError(
            "cann DeepSeek-V4 sparse attention does not support output_attentions=True; use the eager implementation."
        )
    if return_lse:
        raise ValueError(
            "cann DeepSeek-V4 sparse attention does not return the softmax LSE the indexer loss needs; "
            "use the eager or tilelang implementation."
        )
    if query_offset != 0 or ori_kv.shape[1] != q.shape[1]:
        raise ValueError(
            "cann DeepSeek-V4 sparse attention requires one KV row per query row; "
            f"got {q.shape[1]} queries over {ori_kv.shape[1]} rows at query_offset={query_offset}."
        )

    slices = None
    if packed_sequence_slices is not None:
        slices = tuple((int(start), int(end)) for start, end in packed_sequence_slices)
    topk_indices = None if candidates is None else candidates.topk_indices
    if topk_indices is not None and slices is not None and len(slices) > 1:
        segment_starts = _segment_compressed_starts(slices, compress_ratio, q.device)
        topk_indices = torch.where(topk_indices >= 0, topk_indices - segment_starts, -1)

    return npu_sparse_flash_mla(
        q.contiguous(),
        ori_kv.unsqueeze(2).contiguous(),
        cmp_kv.unsqueeze(2).contiguous(),
        topk_indices,
        softmax_scale=sm_scale,
        cmp_ratio=compress_ratio,
        ori_mask_mode=_ORI_SLIDING_CAUSAL_MASK_MODE,
        cmp_mask_mode=_CMP_SEGMENT_CAUSAL_MASK_MODE,
        ori_win_left=sliding_window - 1,
        ori_win_right=0,
        sinks=attn_sink.to(dtype=torch.float32),
        packed_sequence_slices=slices,
    )
