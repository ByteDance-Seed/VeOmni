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

"""Qwen4-Exp QSA block selection on Ascend CANN (``cann_ops_transformer`` LightningIndexerV2).

The ``qsa_indexer`` / ``standard`` contract takes the already pooled, RMS-normed
and RoPE-rotated compressed keys ``[B, N, D]`` and index queries ``[B, S, H, D]``
(any strides, BSND logical layout) and returns per-row selected compressed-block
ids ``[B, S, K]`` with ``-1`` padding. The caller owns pooling/norm/RoPE, the
incomplete-tail append, and the token expansion, so this row stays a drop-in
replacement for the eager ``einsum + ReLU + topk`` selection loop.

CANN encodes visibility as a mask mode instead of per-query ranges. Mode 3 lets
query row ``t`` of a batch element see compressed entries
``[0, (t + 1) // compress_ratio)`` of that element — exactly the eager path's
fully-covered-block causality. Two consequences the eager loop also relies on:

- the row's own incomplete tail block is outside the candidate set (the model
  appends it separately, so ids never duplicate);
- candidate blocks are those the eager path would also score, so selection
  differences come only from kernel rounding, not from masking.

The eager score is ``relu(q . k).sum(heads) / sqrt(head_dim)``. The kernel takes
per-head weights ``[B, S, H]`` and applies them inside its head reduction; the
uniform ``1/sqrt(head_dim)`` weight below reproduces the eager scaling exactly
if the kernel's reduction is ReLU-sum. If it is not, the scores parity test in
``tests/ops/qsa_indexer`` is the gate that catches it — check it before relying
on routing parity. The kernel is forward-only and returns no gradient; the QSA
selection is a discrete token routing, matching the eager path where top-k
indices carry no gradient either.

Shape/plan tensors are cached with a bounded ``lru_cache`` (max 8 plans), keyed
including device, so long decode-style growth cannot accumulate metadata
forever.
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
    residual = torch.full((batch,), seq_len % compress_ratio, dtype=torch.int32, device=device)
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
    """Return ``(indices, scores)`` of the selected compressed blocks.

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


def wrapper(
    index_q: Tensor,
    compressed_keys: Tensor,
    compress_ratio: int,
    top_k: int,
) -> Tensor:
    """Select compressed-KV blocks with the CANN Lightning Indexer.

    ``index_q`` is ``[B, S, H, D]``, ``compressed_keys`` is ``[B, N, D]``, in
    any strides, dtype ``float16``/``bfloat16`` (anything else is cast to
    ``bfloat16``). Returns ``int32 [B, S, top_k]`` block ids; unselectable
    slots hold ``-1``. The selection carries no gradient.
    """
    indices, _ = select_blocks(index_q, compressed_keys, compress_ratio, top_k)
    return indices
