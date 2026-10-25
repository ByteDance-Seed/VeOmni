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

"""DeepSeek-V4 lighting indexer on Ascend CANN (``cann_ops_transformer``).

The ``dsa_indexer`` / ``deepseek_v4`` contract takes SBHD operands and returns
``(scores, indices)``. CANN reads BSND, or TND for a packed batch, and returns
``(indices, values)``. The model passes SBHD views of BSND tensors, so the
transposes below hand CANN the original storage without a copy.

CANN encodes visibility as a mask mode instead of per-query ranges. Mode 3
lets query row ``t`` of a segment see compressed entries
``[0, (t + 1) // compress_ratio)`` of that segment, which is the contract's
default when ``cu_seqlen_ks`` / ``cu_seqlen_ke`` are absent and, per packed
segment, the ranges those segments imply when their positions restart at 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import Tensor


_SEGMENT_CAUSAL_MASK_MODE = 3


def _ops():
    from cann_ops_transformer import ops

    return ops


@lru_cache(maxsize=8)
def _causal_plan(
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


@dataclass(frozen=True)
class _PackedPlan:
    """TND metadata and the per-query ranges one set of packed boundaries implies."""

    compressed_len: int
    top_k: int
    range_starts: Tensor
    range_ends: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    residual: Tensor
    metadata: Tensor | None


@lru_cache(maxsize=8)
def _packed_plan(
    slices: tuple[tuple[int, int], ...],
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    width: int,
    device: torch.device,
) -> _PackedPlan:
    """Build TND metadata and the ranges it implies from host-side boundaries only."""
    lengths = [end - start for start, end in slices]
    compressed = [length // compress_ratio for length in lengths]
    seq_len = slices[-1][1]
    top_k = min(width, max(compressed))
    cu_seqlens_q = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, *accumulate(compressed)], dtype=torch.int32, device=device)
    residual = torch.tensor([length % compress_ratio for length in lengths], dtype=torch.int32, device=device)

    repeats = torch.tensor(lengths, dtype=torch.long, device=device)
    segment_q = torch.repeat_interleave(cu_seqlens_q[:-1], repeats, output_size=seq_len)
    range_starts = torch.repeat_interleave(cu_seqlens_k[:-1], repeats, output_size=seq_len)
    rows = torch.arange(seq_len, dtype=torch.int32, device=device)
    range_ends = range_starts + (rows - segment_q + 1) // compress_ratio

    metadata = None
    if len(slices) > 1:
        metadata = _ops().lightning_indexer_metadata(
            num_heads,
            1,
            head_dim,
            top_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=residual,
            batch_size=len(slices),
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(compressed),
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
        )
    return _PackedPlan(
        compressed_len=sum(compressed),
        top_k=top_k,
        range_starts=range_starts,
        range_ends=range_ends,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        residual=residual,
        metadata=metadata,
    )


def _select_causal(q: Tensor, k: Tensor, weights: Tensor, top_k: int, compress_ratio: int) -> tuple[Tensor, Tensor]:
    batch, seq_len, num_heads, head_dim = q.shape
    compressed_len = k.shape[1]
    if compressed_len != seq_len // compress_ratio:
        raise ValueError(
            "dsa_indexer deepseek_v4 cann derives causal visibility from the query row and needs "
            f"seq_len // compress_ratio compressed entries; got {compressed_len} for seq_len={seq_len}, "
            f"compress_ratio={compress_ratio}"
        )
    residual, metadata = _causal_plan(
        num_heads, head_dim, top_k, batch, seq_len, compressed_len, compress_ratio, q.device
    )
    indices, values = _ops().lightning_indexer(
        q.to(torch.bfloat16).contiguous(),
        k.to(torch.bfloat16).unsqueeze(2).contiguous(),
        weights.float().contiguous(),
        top_k,
        cmp_residual_k=residual,
        metadata=metadata,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
        cmp_ratio=compress_ratio,
        return_value=1,
    )
    return indices.reshape(batch, seq_len, top_k), values.reshape(batch, seq_len, top_k)


def _select_packed(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    width: int,
    compress_ratio: int,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None,
) -> tuple[Tensor, Tensor]:
    batch, seq_len, num_heads, head_dim = q.shape
    if packed_sequence_slices is None:
        raise ValueError(
            "dsa_indexer deepseek_v4 cann takes query visibility from the query row or from "
            "packed_sequence_slices; explicit cu_seqlen_ks/cu_seqlen_ke without slices (context or "
            "Ulysses parallelism, decode, non-canonical position_ids) needs the eager or tilelang implementation"
        )
    slices = tuple((int(start), int(end)) for start, end in packed_sequence_slices)
    if batch != 1 or not slices or slices[0][0] != 0 or slices[-1][1] != seq_len:
        raise ValueError(
            "dsa_indexer deepseek_v4 cann needs batch size 1 and packed_sequence_slices spanning all "
            f"{seq_len} query rows; got batch={batch}, slices={slices}"
        )
    if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            f"Packed indexer ranges must have shape ({seq_len},), got "
            f"{tuple(cu_seqlen_ks.shape)} and {tuple(cu_seqlen_ke.shape)}"
        )
    plan = _packed_plan(slices, compress_ratio, num_heads, head_dim, width, q.device)
    if plan.compressed_len != k.shape[1]:
        raise ValueError(f"packed_sequence_slices imply {plan.compressed_len} compressed entries, got {k.shape[1]}")
    # Positions are not part of the contract; this is the one check that syncs with the host.
    if not bool(((cu_seqlen_ks == plan.range_starts) & (cu_seqlen_ke == plan.range_ends)).all()):
        raise ValueError(
            "dsa_indexer deepseek_v4 cann masks every packed segment causally from its first row, so "
            "cu_seqlen_ks/cu_seqlen_ke must be the ranges packed_sequence_slices imply (position_ids "
            "restarting at 0 in every segment); use the eager or tilelang implementation otherwise"
        )
    if plan.metadata is None:
        return _select_causal(q, k, weights, width, compress_ratio)

    indices, values = _ops().lightning_indexer(
        q[0].to(torch.bfloat16).contiguous(),
        k[0].to(torch.bfloat16).unsqueeze(1).contiguous(),
        weights[0].float().contiguous(),
        plan.top_k,
        cu_seqlens_q=plan.cu_seqlens_q,
        cu_seqlens_k=plan.cu_seqlens_k,
        cmp_residual_k=plan.residual,
        metadata=plan.metadata,
        layout_q="TND",
        layout_k="TND",
        mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
        cmp_ratio=compress_ratio,
        return_value=1,
    )
    indices = indices.reshape(1, seq_len, plan.top_k)
    # TND indices address each segment's own compressed rows; the contract addresses
    # the packed buffer. ``output_idx_offset`` would do this on the device, but the
    # supported Atlas products reject it.
    indices = torch.where(indices >= 0, indices + plan.range_starts.view(1, -1, 1), indices)
    return indices, values.reshape(1, seq_len, plan.top_k)


def wrapper(
    index_q: Tensor,
    index_k: Tensor,
    weights: Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: Tensor | None = None,
    cu_seqlen_ks: Tensor | None = None,
    cu_seqlen_ke: Tensor | None = None,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Select compressed-KV entries with the CANN Lightning Indexer.

    ``index_q`` is ``[S, B, H, D]``, ``index_k`` is ``[S_kv, B, D]``,
    ``weights`` is ``[S, B, H]``, in any strides. Returns scores and indices
    with shape ``[B, S, min(topk, S_kv)]``. Unselectable slots hold index -1
    and score ``-inf``. The kernel is forward-only, so scores carry no gradient.
    """
    if topk_indices is not None:
        raise NotImplementedError(
            "dsa_indexer deepseek_v4 cann selects its own top-k; scoring supplied topk_indices "
            "needs the eager or tilelang implementation"
        )
    if (cu_seqlen_ks is None) != (cu_seqlen_ke is None):
        raise ValueError("cu_seqlen_ks and cu_seqlen_ke must be provided together")

    seq_len, batch = index_q.shape[:2]
    compressed_len = index_k.shape[0]
    q = index_q.transpose(0, 1)
    k = index_k.transpose(0, 1)
    w = weights.transpose(0, 1)
    if compressed_len == 0:
        empty = (batch, seq_len, 0)
        return q.new_empty(empty, dtype=torch.float32), q.new_empty(empty, dtype=torch.int32)

    width = min(topk, compressed_len)
    if cu_seqlen_ks is None:
        indices, values = _select_causal(q, k, w, width, compress_ratio)
    else:
        indices, values = _select_packed(
            q, k, w, width, compress_ratio, cu_seqlen_ks, cu_seqlen_ke, packed_sequence_slices
        )

    indices = indices.to(torch.int32)
    scores = values.float().masked_fill(indices < 0, float("-inf"))
    # TND caps K at the longest segment's compressed length.
    if indices.shape[-1] < width:
        pad = width - indices.shape[-1]
        indices = F.pad(indices, (0, pad), value=-1)
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    return scores, indices
