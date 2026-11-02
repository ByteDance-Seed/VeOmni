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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen4-Exp QSA block selection on Ascend CANN (``cann_ops_transformer`` LightningIndexerV2).

Two call contracts share this module:

- Single-segment batches: ``[B, S, H, D]`` queries with ``[B, N, D]`` pooled
  keys (BSND layout, ``N == S // compress_ratio``). CANN encodes visibility as
  mask mode 3: query row ``t`` of a batch element sees compressed entries
  ``[0, (t + 1) // compress_ratio)`` of that element — the eager path's
  fully-covered-block causality. The row's own incomplete tail block stays
  outside the candidate set (the model appends it separately).
- Flat packed batches (the padding-free training layout): ``[B, S, H, D]``
  queries flattened to ``T = B * S`` tokens, ``[N, D]`` segment-major pooled
  keys, plus the segment boundary tensors. The TND layout carries
  ``cu_seqlens_q``/``cu_seqlens_k`` natively, so the kernel reproduces the
  eager loop's per-segment floor-causal selection for every segment; the
  kernel returns per-segment block ids, which this module offsets into the
  flat segment-major block list (a ``searchsorted`` + vector add).

The eager score is ``relu(q . k).sum(heads) / sqrt(head_dim)``. The kernel
takes per-head weights and applies them inside its head reduction; the
uniform ``1/sqrt(head_dim)`` weight reproduces the eager scaling if the
kernel's reduction is ReLU-sum (probe-verified: index sets match the eager
reference on both layouts; returned values can diverge on a minority of
slots, and the model consumes only ids). The kernel is forward-only and
returns no gradient; the QSA selection is a discrete token routing, matching
the eager path where top-k indices carry no gradient either.

The BSND ``_plan``/metadata is cached with a bounded ``lru_cache``; the TND
plan depends on the per-batch segment boundaries and is rebuilt per call.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


_SEGMENT_CAUSAL_MASK_MODE = 3


def _ops():
    from cann_ops_transformer import ops

    return ops


@lru_cache(maxsize=8)
def _plan(
    num_heads: int,
    head_dim: int,
    top_k: int,
    batch: int,
    seq_len: int,
    compressed_len: int,
    compress_ratio: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    # CANN takes the pre-compression remainder, not the compressed length.
    # LightningIndexerV2 rejects cmp_residual_k outright when cmp_ratio is 1
    # (nothing is compressed, so no remainder exists).
    residual = (
        None
        if compress_ratio == 1
        else torch.full((batch,), seq_len % compress_ratio, dtype=torch.int32, device=device)
    )
    metadata = _ops().lightning_indexer_metadata(
        num_heads,
        1,
        head_dim,
        top_k,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=seq_len,
        max_seqlen_k=compressed_len,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
        cmp_ratio=compress_ratio,
    )
    return residual, metadata


@lru_cache(maxsize=4)
def _head_scale(batch: int, seq_len: int, num_heads: int, head_dim: int, device: torch.device) -> Tensor:
    return torch.full((batch, seq_len, num_heads), head_dim**-0.5, dtype=torch.float32, device=device)


def select_blocks(
    index_q: Tensor,
    compressed_keys: Tensor,
    compress_ratio: int,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """Return ``(indices, scores)`` of the selected compressed blocks (BSND).

    ``index_q`` is ``[B, S, H, D]`` and ``compressed_keys`` is ``[B, N, D]``
    with ``N == S // compress_ratio`` (single-segment causal, no KV cache), in
    any strides. ``indices`` is ``int32 [B, S, top_k]`` with ``-1`` in
    unselectable slots; ``scores`` is ``float32 [B, S, top_k]`` with ``-inf``
    where the index is ``-1``. ``top_k`` is capped at ``N``.
    """
    if index_q.ndim != 4 or compressed_keys.ndim != 3:
        raise ValueError(
            f"qsa_indexer cann takes [B,S,H,D] queries and [B,N,D] compressed keys; got {tuple(index_q.shape)} and {tuple(compressed_keys.shape)}"
        )
    batch, seq_len, num_heads, head_dim = index_q.shape
    compressed_len = compressed_keys.shape[1]
    if compressed_keys.shape[0] != batch or compressed_keys.shape[2] != head_dim:
        raise ValueError("qsa_indexer cann query/key batch and head dimensions must match")
    if compressed_len != seq_len // compress_ratio:
        raise ValueError(
            "qsa_indexer cann derives causal visibility from the query row and needs "
            f"seq_len // compress_ratio compressed entries; got {compressed_len} for seq_len={seq_len}, "
            f"compress_ratio={compress_ratio}"
        )
    if top_k < 1:
        raise ValueError(f"qsa_indexer cann needs top_k >= 1; got {top_k}")
    width = min(top_k, compressed_len)

    with torch.no_grad():
        residual, metadata = _plan(
            num_heads, head_dim, width, batch, seq_len, compressed_len, compress_ratio, index_q.device
        )
        weights = _head_scale(batch, seq_len, num_heads, head_dim, index_q.device)
        indices, values = _ops().lightning_indexer(
            index_q.to(torch.bfloat16).contiguous(),
            compressed_keys.to(torch.bfloat16).unsqueeze(2).contiguous(),
            weights.contiguous(),
            width,
            cmp_residual_k=residual,
            metadata=metadata,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
            return_value=1,
        )

    indices = indices.reshape(batch, seq_len, width).to(torch.int32)
    scores = values.reshape(batch, seq_len, width).float()
    if indices.shape[-1] < top_k:
        pad = top_k - indices.shape[-1]
        indices = torch.nn.functional.pad(indices, (0, pad), value=-1)
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.masked_fill(indices < 0, float("-inf"))
    return indices, scores


def select_blocks_varlen(
    index_q: Tensor,
    compressed_keys: Tensor,
    cu_seq_lens_q: Tensor,
    cu_seq_lens_k: Tensor,
    cmp_residual_k: Tensor,
    compress_ratio: int,
    top_k: int,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Tensor:
    """Return selected compressed-block ids for a flat packed batch (TND).

    ``index_q`` is ``[B, S, H, D]`` (flattened internally to ``B * S`` tokens)
    and ``compressed_keys`` is ``[N, D]`` pooled keys in segment-major order —
    the flat order of the caller's block list. ``cu_seq_lens_q`` /
    ``cu_seq_lens_k`` are ``int32 [b+1]`` device tensors of cumulative token
    and block counts (first element 0, last element ``B * S`` / ``N``);
    ``cmp_residual_k`` is ``int32 [b]`` with ``segment_len % compress_ratio``.
    Each segment is selected with its own floor-causal rule, matching the
    eager segment-causal loop exactly.

    Returns ``int32 [B, S, min(top_k, N)]`` global block ids into the flat
    block list with ``-1`` padding, mirroring the eager loop's
    ``min(top_k, total_blocks)`` width.
    """
    if index_q.ndim != 4 or compressed_keys.ndim != 2:
        raise ValueError(
            "qsa_indexer cann varlen takes [B,S,H,D] queries and flat [N,D] compressed keys; "
            f"got {tuple(index_q.shape)} and {tuple(compressed_keys.shape)}"
        )
    batch, seq_len, num_heads, head_dim = index_q.shape
    total_tokens = batch * seq_len
    total_blocks = compressed_keys.shape[0]
    if compressed_keys.shape[1] != head_dim:
        raise ValueError("qsa_indexer cann varlen key head dimension must match the queries")
    if cu_seq_lens_q.ndim != 1 or cu_seq_lens_k.ndim != 1 or cmp_residual_k.ndim != 1:
        raise ValueError("qsa_indexer cann varlen boundary tensors must be 1-D")
    if cu_seq_lens_q.numel() != cu_seq_lens_k.numel() or cu_seq_lens_q.numel() < 2:
        raise ValueError(
            "qsa_indexer cann varlen needs matching [b+1] cu_seq_lens_q/cu_seq_lens_k; "
            f"got {cu_seq_lens_q.numel()} and {cu_seq_lens_k.numel()}"
        )
    if cmp_residual_k.numel() != cu_seq_lens_q.numel() - 1:
        raise ValueError(
            "qsa_indexer cann varlen needs one cmp_residual_k entry per segment; "
            f"got {cmp_residual_k.numel()} for {cu_seq_lens_q.numel() - 1} segments"
        )
    if top_k < 1:
        raise ValueError(f"qsa_indexer cann needs top_k >= 1; got {top_k}")
    if batch_size is None:
        batch_size = cu_seq_lens_q.numel() - 1
    if max_seqlen_q is None:
        max_seqlen_q = -1
    if max_seqlen_k is None:
        max_seqlen_k = -1
    width = min(top_k, total_blocks)
    if width < 1:
        return torch.full((batch, seq_len, top_k), -1, dtype=torch.int32, device=index_q.device)

    device = index_q.device
    with torch.no_grad():
        # LightningIndexerV2 rejects cmp_residual_k when cmp_ratio is 1; the
        # caller's all-zero entries are dropped for that case.
        kernel_residual = cmp_residual_k if compress_ratio > 1 else None
        # The TND plan encodes the per-segment task split, so it is rebuilt
        # per call (the boundaries change with every batch).
        metadata = _ops().lightning_indexer_metadata(
            num_heads,
            1,
            head_dim,
            width,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            cmp_residual_k=kernel_residual,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
        )
        weights = torch.full((total_tokens, num_heads), head_dim**-0.5, dtype=torch.float32, device=device)
        indices, _ = _ops().lightning_indexer(
            index_q.to(torch.bfloat16).reshape(total_tokens, num_heads, head_dim).contiguous(),
            compressed_keys.to(torch.bfloat16).reshape(total_blocks, 1, head_dim).contiguous(),
            weights,
            width,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            cmp_residual_k=kernel_residual,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
            return_value=0,
        )

    indices = indices.reshape(total_tokens, width).to(torch.int32)
    # The kernel returns per-segment block ids; offset them into the flat
    # segment-major block list (invalid -1 slots stay -1).
    token_segment = torch.searchsorted(
        cu_seq_lens_q, torch.arange(total_tokens, device=device, dtype=cu_seq_lens_q.dtype), right=True
    ) - 1
    segment_offsets = cu_seq_lens_k[:-1].index_select(0, token_segment).view(total_tokens, 1)
    indices = torch.where(indices >= 0, indices + segment_offsets, indices)
    return indices.reshape(batch, seq_len, width)


def wrapper(
    index_q: Tensor,
    compressed_keys: Tensor,
    compress_ratio: int,
    top_k: int,
    *,
    cu_seq_lens_q: Tensor | None = None,
    cu_seq_lens_k: Tensor | None = None,
    cmp_residual_k: Tensor | None = None,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Tensor:
    """Select compressed-KV blocks with the CANN Lightning Indexer.

    Single-segment batches pass ``[B, S, H, D]`` queries with ``[B, N, D]``
    compressed keys, in any strides, dtype ``float16``/``bfloat16`` (anything
    else is cast to ``bfloat16``). Flat packed batches pass the flat ``[N, D]``
    segment-major compressed keys together with the segment boundary tensors
    (see :func:`select_blocks_varlen`). Returns ``int32 [B, S, top_k]`` block
    ids (varlen: width capped at the block count, matching the eager loop);
    unselectable slots hold ``-1``. The selection carries no gradient.
    """
    if cu_seq_lens_q is not None or cu_seq_lens_k is not None or cmp_residual_k is not None:
        if cu_seq_lens_q is None or cu_seq_lens_k is None or cmp_residual_k is None:
            raise ValueError(
                "qsa_indexer cann varlen needs cu_seq_lens_q, cu_seq_lens_k and cmp_residual_k together"
            )
        return select_blocks_varlen(
            index_q,
            compressed_keys,
            cu_seq_lens_q,
            cu_seq_lens_k,
            cmp_residual_k,
            compress_ratio,
            top_k,
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
        )
    indices, _ = select_blocks(index_q, compressed_keys, compress_ratio, top_k)
    return indices
