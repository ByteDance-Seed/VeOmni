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

from itertools import accumulate
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


def zigzag_block_order(cp_size: int) -> List[int]:
    """Return the balanced causal block order for context-parallel ranks."""
    order = []
    for rank in range(cp_size):
        order.append(rank)
        order.append(2 * cp_size - 1 - rank)
    return order


def zigzag_reorder(tensor: Tensor, dim: int, cp_size: int) -> Tensor:
    """Arrange a sequence so each CP rank receives its low and high blocks."""
    if cp_size <= 1:
        return tensor
    num_blocks = 2 * cp_size
    seq_len = tensor.size(dim)
    assert seq_len % num_blocks == 0, f"seq len {seq_len} not divisible by 2*cp_size ({num_blocks})"
    blocks = list(torch.tensor_split(tensor, num_blocks, dim=dim))
    return torch.cat([blocks[index].contiguous() for index in zigzag_block_order(cp_size)], dim=dim)


def zigzag_undo(tensor: Tensor, dim: int, cp_size: int) -> Tensor:
    """Restore a sequence from the context-parallel zig-zag block order."""
    if cp_size <= 1:
        return tensor
    num_blocks = 2 * cp_size
    seq_len = tensor.size(dim)
    assert seq_len % num_blocks == 0, f"seq len {seq_len} not divisible by 2*cp_size ({num_blocks})"
    blocks = list(torch.tensor_split(tensor, num_blocks, dim=dim))
    inverse = [0] * num_blocks
    for new_position, original_position in enumerate(zigzag_block_order(cp_size)):
        inverse[original_position] = new_position
    return torch.cat([blocks[inverse[index]].contiguous() for index in range(num_blocks)], dim=dim)


def zigzag_reorder_packed(tensor: Tensor, cu_seqlens: Tensor, dim: int, cp_size: int) -> Tensor:
    """Arrange every packed sequence independently in zig-zag block order."""
    if cp_size <= 1:
        return tensor
    cumulative_lengths = [int(value) for value in cu_seqlens.tolist()]
    num_blocks = 2 * cp_size
    document_blocks = []
    for start, end in zip(cumulative_lengths[:-1], cumulative_lengths[1:]):
        seq_len = end - start
        assert seq_len % num_blocks == 0, (
            f"document length {seq_len} not divisible by 2*cp_size ({num_blocks}); "
            "packed varlen under context-parallel requires every document length "
            "to be a multiple of 2*cp_size"
        )
        document = tensor.narrow(dim, start, seq_len)
        document_blocks.append(list(torch.tensor_split(document, num_blocks, dim=dim)))

    pieces = []
    for rank in range(cp_size):
        for blocks in document_blocks:
            pieces.append(blocks[rank])
            pieces.append(blocks[num_blocks - 1 - rank])
    return torch.cat([piece.contiguous() for piece in pieces], dim=dim)


def local_cu_seqlens(cu_seqlens: Tensor, cp_size: int) -> Tensor:
    """Build per-rank packed-sequence offsets after zig-zag slicing."""
    if cp_size <= 1:
        return cu_seqlens
    cumulative_lengths = [int(value) for value in cu_seqlens.tolist()]
    num_blocks = 2 * cp_size
    local_lengths = [0]
    for start, end in zip(cumulative_lengths[:-1], cumulative_lengths[1:]):
        seq_len = end - start
        assert seq_len % num_blocks == 0, (
            f"document length {seq_len} not divisible by 2*cp_size ({num_blocks}); "
            "packed varlen under context-parallel requires every document length "
            "to be a multiple of 2*cp_size"
        )
        local_lengths.append(local_lengths[-1] + seq_len // cp_size)
    return torch.tensor(local_lengths, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


# Compatibility name for callers that use the FlashAttention terminology.
zigzag_reorder_varlen = zigzag_reorder_packed


def usp_zigzag_slice(
    tensor: Tensor,
    doc_cu_seqlens: Optional[Union[Tensor, Sequence[int]]],
    dim: int,
    cp_size: int,
    cp_rank: int,
    ulysses_size: int,
    ulysses_rank: int,
) -> Tensor:
    """Take this rank's USP shard of a text-axis tensor: cp outer (zig-zag), ulysses inner (contiguous).

    ``doc_cu_seqlens`` are the document offsets on the full (padded) sequence. With more than one
    document every document is zig-zag split independently; otherwise the whole sequence is one
    document. The collator slices ``input_ids`` / ``labels`` / ``position_ids`` with this function
    and the model slices the multimodal placeholder masks with it, so both share one layout.
    """
    if cp_size <= 1:
        return tensor
    if doc_cu_seqlens is not None and len(doc_cu_seqlens) > 2:
        if not isinstance(doc_cu_seqlens, Tensor):
            doc_cu_seqlens = torch.tensor(list(doc_cu_seqlens), dtype=torch.int64)
        reordered = zigzag_reorder_packed(tensor, doc_cu_seqlens, dim=dim, cp_size=cp_size)
    else:
        reordered = zigzag_reorder(tensor, dim=dim, cp_size=cp_size)
    cp_chunk = reordered.size(dim) // cp_size
    cp_region = reordered.narrow(dim, cp_rank * cp_chunk, cp_chunk)
    if ulysses_size <= 1:
        return cp_region.contiguous()
    ulysses_chunk = cp_region.size(dim) // ulysses_size
    return cp_region.narrow(dim, ulysses_rank * ulysses_chunk, ulysses_chunk).contiguous()


# --------------------------------------------------------------------------- #
# ViT layout for bidirectional ring attention.                                #
#                                                                             #
# A vit seq is one varlen attention unit of the vision encoder (one image, or #
# one temporal step of a video). Every vit seq is split into ``cp_size``      #
# shards with ``merger_unit`` rows (one spatial-merge window) as the smallest #
# unit, so the patch merger never straddles two ranks; the remainder units go #
# to the cp ranks in global round-robin order so the per-rank totals differ   #
# by at most one unit. A cp rank's shards are concatenated (the cp local seq) #
# and padded at the tail to a common length that is divisible by             #
# ``merger_unit * ulysses_size``; the cp local seq is then split contiguously #
# across the ulysses ranks.                                                   #
# --------------------------------------------------------------------------- #
def _vit_shard_lens(
    vit_seqlens: Sequence[int],
    cp_size: int,
    ulysses_size: int,
    merger_unit: int = 4,
) -> Tuple[List[List[int]], int]:
    """Return ``(shard_lens, cp_local_len)``: rows of vit seq ``i`` on cp rank ``c`` and the padded cp local length."""
    num_seqs = len(vit_seqlens)
    shard_units = [[0] * num_seqs for _ in range(cp_size)]
    start = 0
    for index, seqlen in enumerate(vit_seqlens):
        if seqlen % merger_unit != 0:
            raise ValueError(f"vit seq length {seqlen} is not divisible by merger_unit ({merger_unit})")
        base, remainder = divmod(seqlen // merger_unit, cp_size)
        for rank in range(cp_size):
            shard_units[rank][index] = base + (1 if (rank - start) % cp_size < remainder else 0)
        start = (start + remainder) % cp_size
    max_units = max((sum(units) for units in shard_units), default=0)
    cp_local_units = -(-max_units // ulysses_size) * ulysses_size
    shard_lens = [[units * merger_unit for units in rank_units] for rank_units in shard_units]
    return shard_lens, cp_local_units * merger_unit


def vit_usp_slice(
    tensor: Tensor,
    vit_seqlens: Sequence[int],
    dim: int,
    cp_size: int,
    cp_rank: int,
    ulysses_size: int,
    ulysses_rank: int,
    pad_value: Union[int, float] = 0,
    merger_unit: int = 4,
) -> Tensor:
    """Take this rank's shard of a ViT patch-axis tensor for bidirectional ring attention.

    ``tensor`` holds the full packed patch stream along ``dim`` (``sum(vit_seqlens)`` rows). The
    result is this rank's ulysses block of its cp local seq: the shards of every vit seq owned by
    ``cp_rank`` in order, tail-padded with ``pad_value``. All ranks get the same length.
    """
    dim = dim % tensor.dim()
    total = sum(vit_seqlens)
    if tensor.size(dim) != total:
        raise ValueError(f"tensor has {tensor.size(dim)} rows along dim {dim}, expected sum(vit_seqlens)={total}")
    ulysses_rank = ulysses_rank if ulysses_size > 1 else 0
    shard_lens, cp_local_len = _vit_shard_lens(vit_seqlens, cp_size, ulysses_size, merger_unit)

    # (global_start, length) of this cp rank's shards, in cp local seq order.
    ranges = []
    seq_start = 0
    for index, seqlen in enumerate(vit_seqlens):
        offset = sum(shard_lens[rank][index] for rank in range(cp_rank))
        if shard_lens[cp_rank][index] > 0:
            ranges.append((seq_start + offset, shard_lens[cp_rank][index]))
        seq_start += seqlen

    block = cp_local_len // ulysses_size
    block_start, block_end = ulysses_rank * block, (ulysses_rank + 1) * block
    pieces = []
    local_pos = 0
    for global_start, length in ranges:
        lo, hi = max(local_pos, block_start), min(local_pos + length, block_end)
        if lo < hi:
            pieces.append(tensor.narrow(dim, global_start + lo - local_pos, hi - lo))
        local_pos += length
    pad_len = block_end - max(local_pos, block_start)
    if pad_len > 0:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_len
        pieces.append(torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device))
    return torch.cat(pieces, dim=dim) if len(pieces) > 1 else pieces[0].contiguous()


def vit_cp_cu_seqlens(
    cu_seqlens: Union[Tensor, Sequence[int]],
    cp_size: int,
    cp_rank: int,
    ulysses_size: int,
    merger_unit: int = 4,
) -> List[int]:
    """Return the local ``cu_seqlens`` of ``cp_rank``'s cp local seq for bidirectional ring attention.

    ``cu_seqlens`` are the GLOBAL offsets of the real vit seqs (no padding segment). The result has
    one entry per vit seq plus a trailing padding segment (possibly empty); its last value is the
    cp local length, identical on every cp rank.
    """
    offsets = cu_seqlens.tolist() if isinstance(cu_seqlens, Tensor) else list(cu_seqlens)
    vit_seqlens = [end - start for start, end in zip(offsets[:-1], offsets[1:])]
    shard_lens, cp_local_len = _vit_shard_lens(vit_seqlens, cp_size, ulysses_size, merger_unit)
    lengths = shard_lens[cp_rank] + [cp_local_len - sum(shard_lens[cp_rank])]
    return [0, *accumulate(lengths)]
