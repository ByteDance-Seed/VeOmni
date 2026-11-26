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

Context parallelism hands over the replicated KV buffer plus the row where the
local query shard starts. Because the kernel has no offset input and counts
causality from the KV end, this adapter narrows the buffer to the prefix
``[0, query_offset + q_len)`` before calling it, so the local queries end at the
last row on every rank. That is the same layout MindSpeed-LLM calls prefix KV.
Packed slices stay global, so every sample that intersects the shard becomes its
own single-segment call starting at that sample's first row.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import accumulate

import torch
from torch import Tensor

from ......distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
from ...sparse_indices import CompressedCandidates
from ...vendor.npu_sparse_flash_mla import npu_sparse_flash_mla
from . import eager


_ORI_SLIDING_CAUSAL_MASK_MODE = 4
_CMP_SEGMENT_CAUSAL_MASK_MODE = 3

# ``cmp_topk`` widths the Sparse FlashMLA kernel is built for. The reference
# implementation derives ``min(index_topk, compressed_len)`` and pads that up to the
# configured width with ``-1``, so a selection is only ever the whole compressed
# table or one of these two widths.
_SFMLA_TOPK_WIDTHS = (512, 1024)


def _queries_are_a_context_parallel_shard() -> bool:
    """Whether this call's queries are a CP shard of the replicated KV buffer.

    Shapes alone do not say: a CP rank and a decode call both pass a buffer longer
    than their queries. Only the parallel state does, and Ulysses counts as non-CP
    here because it gathers the sequence back into the queries.
    """
    if not is_parallel_state_initialized():
        return False
    parallel_state = get_parallel_state()
    return parallel_state.cp_enabled and not parallel_state.ulysses_enabled


def _selection_for_local_table(topk_indices: Tensor, compressed_len: int) -> Tensor:
    """Return the selection columns this call serves, dropping producer padding.

    A selection is served when it spans the whole compressed table or matches one of
    the kernel widths. The vendor wrapper pads anything at most 512 wide up to 512,
    so the short-table case (every row selected, ``-1`` after them) reaches the
    kernel in the shape it was built for, and a 512/1024-wide selection is served
    even over a wider table, which is what the shipped ``index_topk: 512`` always
    produces. The narrowed prefix grows with the rank, ``(cp_rank + 1) * L / rate``,
    so rank 0 spans its table exactly when ``L == rate * index_topk`` and later ranks
    exceed it: in the ``cp_size=2`` comparison in
    ``docs/design/deepseek_v4_context_parallel.md`` (``max_seq_len: 4096``, rate 4)
    the two ranks narrow to 512 and 1024 rows, exercising the spanning and the
    wide-table case in the same step, and at the shipped ``max_seq_len: 8192`` both
    ranks are wide-table (1024 and 2048 rows). The 513-row parity case in
    ``tests/ops/dsa/test_dsa_deepseek_v4_cann.py`` covers the width over a table
    wider than the selection on its own.

    A selection below 512 that does not span its table is not served: on CANN 9.2.1
    such a call (16 indices over a 75-row table) kept the eager direction in the
    forward but scaled some rows down and produced a nonfinite ``q`` gradient, so
    the adapter refuses a kernel call with no defined output rather than call it.
    The model only produces this below-512 shape when ``index_topk`` is neither 512
    nor 1024 and the table is wider; ``docs/usage/arguments.md`` states that limit.

    Context parallelism narrows the compressed table to the local prefix while the
    indexer pads its selection to the global width, so a rank can receive more
    columns than it has rows. Those trailing columns are ``-1`` by construction;
    trimming them restores the short-table shape above instead of rejecting a
    selection the producer built correctly. A real index in them addresses a row
    this call does not own, so that stays an error.
    """
    width = topk_indices.shape[-1]
    if width in _SFMLA_TOPK_WIDTHS or width == compressed_len:
        return topk_indices
    if width < compressed_len:
        raise ValueError(
            "cann DeepSeek-V4 sparse attention needs the compressed selection to span the whole table "
            f"or be 512/1024 wide; got {width} indices over {compressed_len} compressed rows."
        )
    if bool((topk_indices[..., compressed_len:] >= 0).any()):
        raise ValueError(
            "cann DeepSeek-V4 sparse attention received a selection wider than the compressed table this "
            f"call serves ({width} indices over {compressed_len} rows) that selects a row past it."
        )
    return topk_indices[..., :compressed_len]


def _standard_layout(x: Tensor) -> Tensor:
    """Return ``x`` with row-major strides and a zero storage offset.

    ``contiguous()`` is a no-op on a packed sample slice or a narrowed prefix: the
    batch is one and the KV head axis is one, and torch ignores the strides of
    size-1 dimensions, so the view keeps the parent buffer's leading stride and
    its storage offset. The aclnn argument check compares strides numerically and
    rejects that view, so it has to be materialized.
    """
    stride = 1
    for size, actual in zip(reversed(x.shape), reversed(x.stride())):
        if actual != stride:
            break
        stride *= size
    else:
        if x.storage_offset() == 0:
            return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out.copy_(x)
    return out


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


def _packed_context_parallel_call(
    q: Tensor,
    ori_kv: Tensor,
    cmp_kv: Tensor,
    attn_sink: Tensor,
    topk_indices: Tensor | None,
    slices: tuple[tuple[int, int], ...],
    query_offset: int,
    *,
    sliding_window: int,
    compress_ratio: int,
    sm_scale: float | None,
) -> Tensor:
    """Run one single-segment call per packed sample that intersects this shard.

    A caller-supplied ``attention_mask`` never reaches here: the wrapper runs the
    eager row for any call that carries one, because the kernel encodes
    visibility as mask modes rather than a per-key mask.

    The packed TND path segments queries and KV alike, so it cannot express the
    prefix rows a local query still needs. Each sample instead becomes its own
    call whose buffer starts at that sample's first row, which is the origin the
    kernel's mask modes count from, so causality and the sliding window stay
    unchanged. Slices are global and this shard covers ``[query_offset, +q_len)``,
    so a sample boundary may fall inside it; the boundary sample is truncated at
    the shard, exactly as the prefix narrowing does for an unpacked sequence.
    """
    compressed_starts: list[int] = []
    compressed_rows = 0
    for start, end in slices:
        compressed_starts.append(compressed_rows)
        compressed_rows += (end - start) // compress_ratio

    outputs = []
    for (start, end), compressed_start in zip(slices, compressed_starts):
        first = max(start, query_offset)
        last = min(end, query_offset + q.shape[1])
        if first >= last:
            continue
        rows = slice(first - query_offset, last - query_offset)
        available = (last - start) // compress_ratio
        segment_topk = None if topk_indices is None else topk_indices[:, rows]
        if segment_topk is not None:
            segment_topk = torch.where(segment_topk >= 0, segment_topk - compressed_start, -1)
        if available == 0:
            # No compressed row is causally visible in this sample (a padding tail,
            # or a sample narrower than the rate): the sliding window is the whole
            # result, which the eager row already implements. The offset rebases the
            # local queries onto the sample-local buffer -- ``position_ids`` only
            # masks visibility, it does not address KV rows. ``cmp_kv`` is passed as
            # a zero-row slice rather than ``None`` because its all-gather is a
            # collective: dropping it here would skip this rank's backward.
            positions = torch.arange(first - start, last - start, device=q.device)
            outputs.append(
                eager.wrapper(
                    q[:, rows],
                    ori_kv[:, start:last],
                    cmp_kv[:, compressed_start:compressed_start],
                    attn_sink,
                    positions.unsqueeze(0).expand(q.shape[0], -1),
                    sliding_window=sliding_window,
                    compress_ratio=compress_ratio,
                    query_offset=first - start,
                    sm_scale=sm_scale,
                )
            )
            continue
        if segment_topk is not None:
            segment_topk = _selection_for_local_table(segment_topk, available)
        seg_q = _standard_layout(q[:, rows])
        seg_kv = _standard_layout(ori_kv[:, start:last].unsqueeze(2))
        seg_cmp = _standard_layout(cmp_kv[:, compressed_start : compressed_start + available].unsqueeze(2))
        outputs.append(
            npu_sparse_flash_mla(
                seg_q,
                seg_kv,
                seg_cmp,
                segment_topk,
                softmax_scale=sm_scale,
                cmp_ratio=compress_ratio,
                cmp_residual_kv=torch.full(
                    (q.shape[0],), (last - start) % compress_ratio, dtype=torch.int32, device=q.device
                ),
                ori_mask_mode=_ORI_SLIDING_CAUSAL_MASK_MODE,
                cmp_mask_mode=_CMP_SEGMENT_CAUSAL_MASK_MODE,
                ori_win_left=sliding_window - 1,
                ori_win_right=0,
                sinks=attn_sink.to(dtype=torch.float32),
            )
        )
    return torch.cat(outputs, dim=1)


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

    Layers without compressed rows run the eager row. The kernel encodes
    visibility as mask modes, not as a per-key mask, so a caller-supplied
    ``attention_mask`` cannot be expressed: any call that carries one runs the
    eager row, packed context-parallel shards included, rather than attending to
    keys the caller excluded. The kernel needs the query
    rows to end at the last KV row: a non-zero ``query_offset`` on a context-parallel
    shard is served by narrowing ``ori_kv`` to that prefix, and packed
    slices then need one call per sample because the TND path segments queries
    and KV alike. A prefix that ends inside the first compression window holds no
    compressed row at all; those queries run the eager row, since the kernel has
    no mask mode for an empty compressed table. A shorter query block that also
    ends at the last KV row is a decode layout, which the kernel does not
    implement and outside context parallelism the adapter refuses. It validates
    its own dtype,
    window, ratio and head-size limits, and this adapter trims the padding a
    context-parallel call inherits from the producer's global selection width,
    rejecting a selection that neither spans the table nor is one of the kernel
    widths, and one that names a row past it. Packed calls without CP take the TND
    path, which needs batch size 1 and reads the slice boundaries on the host.
    """
    if attention_mask is not None or cmp_kv is None or cmp_kv.shape[1] == 0:
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
    slices = None
    if packed_sequence_slices is not None:
        slices = tuple((int(start), int(end)) for start, end in packed_sequence_slices)
    topk_indices = None if candidates is None else candidates.topk_indices

    # The kernel derives each query's row from ``kv_len - q_len`` and encodes causality
    # as a mask mode, so it cannot express "this shard starts at row ``query_offset``":
    # it assumes the queries end at the last KV row. CP keeps the reproduced buffer
    # global and passes the shard's start, so narrowing to the prefix
    # ``[0, query_offset + q_len)`` restores the assumption on every rank. Causal
    # attention never reads past the last local query, so the result is unchanged; the
    # slice is a view, and only that prefix is copied by the ``contiguous`` below.
    kv_len = query_offset + q.shape[1]
    if query_offset < 0 or ori_kv.shape[1] < kv_len:
        raise ValueError(
            "cann DeepSeek-V4 sparse attention needs the local queries to end inside the KV buffer: "
            f"got {q.shape[1]} queries at query_offset={query_offset} over {ori_kv.shape[1]} rows."
        )
    if not _queries_are_a_context_parallel_shard() and (query_offset != 0 or ori_kv.shape[1] != q.shape[1]):
        # Queries that end at the last KV row while covering fewer rows than the
        # buffer are decode. The kernel numbers queries from ``kv_len - q_len`` and
        # has no cache layout, so the output is undefined; the indexer refuses this
        # one op earlier, and the adapter refuses it too instead of widening its
        # domain silently.
        raise ValueError(
            "cann DeepSeek-V4 sparse attention implements the prefill/training layout only: outside context "
            f"parallelism the queries must be the whole KV buffer (got {q.shape[1]} queries at "
            f"query_offset={query_offset} over {ori_kv.shape[1]} rows). Decode-shaped calls need the eager "
            "implementation."
        )
    # Packed slices describe the global buffer while a CP rank owns a window of it, so
    # they end past the local queries even on the rank whose window starts at row 0.
    if slices is not None and len(slices) > 1 and (query_offset != 0 or slices[-1][1] != q.shape[1]):
        return _packed_context_parallel_call(
            q,
            ori_kv,
            cmp_kv,
            attn_sink,
            topk_indices,
            slices,
            query_offset,
            sliding_window=sliding_window,
            compress_ratio=compress_ratio,
            sm_scale=sm_scale,
        )
    if kv_len < compress_ratio:
        # The last local query ends inside the first compression window, so no
        # compressed row is visible and the sliding window is the whole result.
        # The kernel has no mask mode for an empty compressed table, so run the
        # eager row. The zero-row slice keeps the all-gather in the autograd graph
        # instead of dropping this rank's share of its backward.
        return eager.wrapper(
            q,
            ori_kv[:, :kv_len],
            cmp_kv[:, :0],
            attn_sink,
            position_ids,
            sliding_window=sliding_window,
            compress_ratio=compress_ratio,
            candidates=candidates,
            query_offset=query_offset,
            attention_mask=attention_mask,
            sm_scale=sm_scale,
        )
    del position_ids, attention_mask
    if ori_kv.shape[1] != kv_len:
        ori_kv = ori_kv[:, :kv_len]
        cmp_kv = cmp_kv[:, : kv_len // compress_ratio]

    if topk_indices is not None and slices is not None and len(slices) > 1:
        segment_starts = _segment_compressed_starts(slices, compress_ratio, q.device)
        topk_indices = torch.where(topk_indices >= 0, topk_indices - segment_starts, -1)
    if topk_indices is not None:
        # A TND call indexes each segment's own table, while the producer sizes the
        # selection from the widest segment. The packed total the rows add up to is
        # not a table any query addresses, so validating against it would refuse a
        # selection the indexer built correctly.
        table_len = cmp_kv.shape[1]
        if slices is not None and len(slices) > 1:
            table_len = max((end - start) // compress_ratio for start, end in slices)
        topk_indices = _selection_for_local_table(topk_indices, table_len)

    return npu_sparse_flash_mla(
        _standard_layout(q),
        _standard_layout(ori_kv.unsqueeze(2)),
        _standard_layout(cmp_kv.unsqueeze(2)),
        topk_indices,
        softmax_scale=sm_scale,
        cmp_ratio=compress_ratio,
        # The vendor wrapper would derive the residual from ``q_len``, but the
        # compressed table is aligned to the narrowed KV prefix.
        cmp_residual_kv=torch.full((q.shape[0],), kv_len % compress_ratio, dtype=torch.int32, device=q.device),
        ori_mask_mode=_ORI_SLIDING_CAUSAL_MASK_MODE,
        cmp_mask_mode=_CMP_SEGMENT_CAUSAL_MASK_MODE,
        ori_win_left=sliding_window - 1,
        ori_win_right=0,
        sinks=attn_sink.to(dtype=torch.float32),
        packed_sequence_slices=slices,
    )
