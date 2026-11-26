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
lets query row ``t`` see compressed entries ``[0, (t + 1) // compress_ratio)``,
which is the contract's default when ``cu_seqlen_ks`` / ``cu_seqlen_ke`` are
absent and, per packed segment, the ranges those segments imply when their
positions restart at 0. The kernel numbers a query row from
``kv_len - q_len``, where ``kv_len`` is the *pre-compression* length it
reconstructs as ``compressed_len * compress_ratio + cmp_residual_k``. Context
parallelism exploits that inference instead of an offset input it does not
have: see ``_select_context_parallel``, and ``_select_packed_context_parallel``
for the packed shards that additionally narrow every intersecting sample to
its own compressed rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import Tensor

from ......distributed.parallel_state import get_parallel_state, is_parallel_state_initialized


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
    query_offset: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    # CANN takes the pre-compression remainder, not the compressed length. It
    # reconstructs ``kv_len`` as ``compressed_len * ratio + remainder`` and numbers
    # the first query row ``kv_len - seq_len``, so this is what lifts a shard onto
    # its global rows; off the context-parallel path the offset is zero.
    residual = torch.full((batch,), (query_offset + seq_len) % compress_ratio, dtype=torch.int32, device=device)
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
        num_heads, head_dim, top_k, batch, seq_len, compressed_len, compress_ratio, 0, q.device
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


def _context_parallel_offset(local_seq_len: int) -> int | None:
    """This rank's first global query row, or ``None`` when CP is off.

    ``None`` keeps every other caller of the explicit-range path -- Ulysses,
    decode, non-canonical positions -- on the refusal ``_select_packed`` already
    raises, because none of them shards contiguous ``cp_rank * L`` blocks.
    """
    if not is_parallel_state_initialized():
        return None
    parallel_state = get_parallel_state()
    if not parallel_state.cp_enabled or parallel_state.ulysses_enabled:
        return None
    return parallel_state.cp_rank * local_seq_len


@lru_cache(maxsize=8)
def _context_parallel_range_ends(seq_len: int, query_offset: int, compress_ratio: int, device: torch.device) -> Tensor:
    """The ends canonical global position ids imply for a shard's query rows.

    Cached because the tensor is rebuilt otherwise on every layer, not because
    validating against it can be skipped: ``_select_context_parallel`` compares
    the caller's ranges with it on every call.
    """
    return ((query_offset + torch.arange(seq_len, device=device) + 1) // compress_ratio).to(torch.int32)


def _select_context_parallel(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    width: int,
    compress_ratio: int,
    query_offset: int,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
) -> tuple[Tensor, Tensor]:
    """Select with CANN over the replicated compressed table from one CP shard.

    A CP rank owns queries ``[query_offset, query_offset + q_len)`` while the
    compressed table is all-gathered and global. The kernel has no offset input
    and numbers a query row from ``kv_len - q_len``, so the adapter narrows the
    table to the prefix this shard's last query can see,
    ``(query_offset + q_len) // compress_ratio`` rows, and passes the remainder
    of *that prefix* as ``cmp_residual_k``. The kernel then reconstructs
    ``kv_len = query_offset + q_len`` and starts numbering at ``query_offset``,
    which is exactly the global row of the local first query, so query ``t``
    sees ``[0, (query_offset + t + 1) // compress_ratio)``. Causal selection
    never reads past the last local query, so nothing is recomputed and no
    padding rows are scored; the narrowing is a slice of the gathered buffer, so
    only the canonical copy the operands need shrinks with it.
    """
    batch, seq_len, num_heads, head_dim = q.shape
    prefix_len = (query_offset + seq_len) // compress_ratio
    # A range table of the wrong width would be compared under broadcasting: a
    # zero-row shard of it compares equal to nothing and ``.all()`` then reports
    # agreement, while a batch axis would compare every query with row 0.
    for name, ranges in (("cu_seqlen_ks", cu_seqlen_ks), ("cu_seqlen_ke", cu_seqlen_ke)):
        if ranges.numel() != seq_len or ranges.shape[-1] != seq_len:
            raise ValueError(
                f"dsa_indexer deepseek_v4 cann under context parallelism needs one {name} per local "
                f"query row; got shape {tuple(ranges.shape)} for seq_len={seq_len}"
            )
    # The kernel never receives these ranges: it re-derives visibility from the
    # geometry, so a caller whose ranges disagree would be masked canonically
    # instead of refused. Every call therefore compares them with the ends the
    # global position ids imply; only that expected tensor is cached, because the
    # shard geometry is fixed at launch.
    expected_ends = _context_parallel_range_ends(seq_len, query_offset, compress_ratio, q.device)
    if not bool(((cu_seqlen_ks == 0) & (cu_seqlen_ke == expected_ends)).all()):
        raise ValueError(
            "dsa_indexer deepseek_v4 cann under context parallelism masks each local query from its "
            "global row, so cu_seqlen_ks/cu_seqlen_ke must be the ranges canonical global position_ids "
            f"imply for query rows [{query_offset}, {query_offset + seq_len}); use the eager or tilelang "
            "implementation otherwise"
        )
    if prefix_len == 0:
        empty = (batch, seq_len, 0)
        return q.new_empty(empty, dtype=torch.float32), q.new_empty(empty, dtype=torch.int32)
    top_k = min(width, prefix_len)
    residual, metadata = _causal_plan(
        num_heads,
        head_dim,
        top_k,
        batch,
        seq_len,
        prefix_len,
        compress_ratio,
        query_offset,
        q.device,
    )
    indices, values = _ops().lightning_indexer(
        q.to(torch.bfloat16).contiguous(),
        # ``contiguous()`` is a no-op on this narrowed prefix: torch ignores the
        # strides of the size-1 batch dimension, so the view keeps the gathered
        # buffer's leading stride and offset. The kernel compares strides
        # numerically, so the prefix is materialized into a canonical buffer.
        k[:, :prefix_len].to(torch.bfloat16).clone(memory_format=torch.contiguous_format).unsqueeze(2),
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
            "packed_sequence_slices; explicit cu_seqlen_ks/cu_seqlen_ke without slices (Ulysses "
            "parallelism, decode, non-canonical position_ids) needs the eager or tilelang implementation"
        )
    slices = tuple((int(start), int(end)) for start, end in packed_sequence_slices)
    if batch != 1 or not slices or slices[0][0] != 0 or slices[-1][1] != seq_len:
        raise ValueError(
            "dsa_indexer deepseek_v4 cann needs batch size 1 and packed_sequence_slices spanning all "
            f"{seq_len} query rows; got batch={batch}, slices={slices}. A Ulysses shard keeps the "
            "packed slices global while holding one shard of the queries, so a packed Ulysses run "
            "needs the eager or tilelang implementation"
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


def _select_packed_context_parallel(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    width: int,
    compress_ratio: int,
    query_offset: int,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    packed_sequence_slices: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    """Select with CANN over the replicated packed table from one CP shard.

    The packed TND path segments queries and KV alike, so it cannot express the
    prefix rows a local query still needs -- a packed sample may start before
    the shard. Each sample that intersects the shard instead becomes its own
    BSND call whose compressed buffer starts at that sample's first compressed
    row, which is the origin the kernel's causal mask counts from. The kernel
    has no offset input and numbers a query row from ``kv_len - q_len``, so the
    call passes the sample's prefix ``(last - start) // compress_ratio``
    compressed rows with the remainder of that prefix as ``cmp_residual_k``: it
    reconstructs ``kv_len = last - start`` and numbers the shard's first query
    of the sample at its row within the sample, which restores the packed
    causal range exactly. A sample boundary may fall inside the shard; the
    boundary sample is truncated at the shard, exactly as the prefix narrowing
    does for an unpacked sequence.
    """
    batch, seq_len, num_heads, head_dim = q.shape
    if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            f"Packed indexer ranges must have shape ({seq_len},), got "
            f"{tuple(cu_seqlen_ks.shape)} and {tuple(cu_seqlen_ke.shape)}"
        )
    slices = tuple((int(start), int(end)) for start, end in packed_sequence_slices)
    plan = _packed_plan(slices, compress_ratio, num_heads, head_dim, width, q.device)
    if plan.compressed_len != k.shape[1]:
        raise ValueError(f"packed_sequence_slices imply {plan.compressed_len} compressed entries, got {k.shape[1]}")
    # The compressed table is replicated, so the ranges stay global and must be
    # the ones the rows this shard owns imply. A rank whose ranges were built
    # for another shard would still run, and mask from the wrong origin.
    if not bool(
        (
            (cu_seqlen_ks == plan.range_starts[query_offset : query_offset + seq_len])
            & (cu_seqlen_ke == plan.range_ends[query_offset : query_offset + seq_len])
        ).all()
    ):
        raise ValueError(
            "dsa_indexer deepseek_v4 cann masks every packed sample causally from its first row, so "
            "cu_seqlen_ks/cu_seqlen_ke must be the global ranges the local query rows' samples imply; "
            "use the eager or tilelang implementation otherwise"
        )

    compressed_starts: list[int] = []
    compressed_rows = 0
    for start, end in slices:
        compressed_starts.append(compressed_rows)
        compressed_rows += (end - start) // compress_ratio

    indices_parts: list[Tensor] = []
    values_parts: list[Tensor] = []
    for (start, end), compressed_start in zip(slices, compressed_starts):
        first = max(start, query_offset)
        last = min(end, query_offset + seq_len)
        if first >= last:
            continue
        rows = slice(first - query_offset, last - query_offset)
        available = (last - start) // compress_ratio
        if available == 0:
            # No compressed row is causally visible in this sample (a sample
            # narrower than the rate, or the shard's head of one): every
            # selectable slot stays -1 / -inf, which the eager row yields too.
            indices_parts.append(q.new_full((batch, last - first, width), -1, dtype=torch.int32))
            values_parts.append(q.new_full((batch, last - first, width), float("-inf"), dtype=torch.float32))
            continue
        top_k = min(width, available)
        residual, metadata = _causal_plan(
            num_heads,
            head_dim,
            top_k,
            batch,
            last - first,
            available,
            compress_ratio,
            first - start,
            q.device,
        )
        # The kernel rejects views: a mid-table slice keeps its storage offset
        # and a sample slice keeps the parent's row stride, which the kernel reads
        # as addressing on the wrong axis. ``.contiguous()`` does not help -- torch
        # ignores the strides of size-1 dimensions, so a batch of one is already
        # "contiguous" and the offset survives -- and neither does a plain
        # ``.clone()``, which copies the layout as-is. Every operand is materialized
        # into a fresh canonical buffer instead; the copies cost one segment-sized
        # buffer per call.
        indices, values = _ops().lightning_indexer(
            q[:, rows].to(torch.bfloat16).clone(memory_format=torch.contiguous_format),
            k[:, compressed_start : compressed_start + available]
            .to(torch.bfloat16)
            .clone(memory_format=torch.contiguous_format)
            .unsqueeze(2),
            weights[:, rows].float().clone(memory_format=torch.contiguous_format),
            top_k,
            cmp_residual_k=residual,
            metadata=metadata,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
            return_value=1,
        )
        indices = indices.reshape(batch, last - first, top_k)
        values = values.reshape(batch, last - first, top_k)
        # The kernel's selection addresses the sample's own compressed rows; the
        # contract addresses the replicated packed buffer. ``output_idx_offset``
        # would do this on the device, but the supported Atlas products reject it.
        indices = torch.where(indices >= 0, indices + compressed_start, indices)
        pad = width - top_k
        if pad:
            indices = F.pad(indices, (0, pad), value=-1)
            values = F.pad(values, (0, pad), value=float("-inf"))
        indices_parts.append(indices)
        values_parts.append(values)
    return torch.cat(indices_parts, dim=1), torch.cat(values_parts, dim=1)


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
        # Context parallelism supplies global ranges over its local query shard.
        # A packed CP shard also keeps global slices, which the single TND call
        # cannot express, so it takes one call per packed sample instead;
        # everything else with explicit ranges falls through to the packed path,
        # which refuses the shapes it cannot express.
        query_offset = _context_parallel_offset(seq_len)
        if query_offset is not None and packed_sequence_slices is not None:
            indices, values = _select_packed_context_parallel(
                q, k, w, width, compress_ratio, query_offset, cu_seqlen_ks, cu_seqlen_ke, packed_sequence_slices
            )
        elif query_offset is not None:
            indices, values = _select_context_parallel(
                q, k, w, width, compress_ratio, query_offset, cu_seqlen_ks, cu_seqlen_ke
            )
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
