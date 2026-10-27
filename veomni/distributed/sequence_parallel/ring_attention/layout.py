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

from typing import List

import torch
from torch import Tensor


def zigzag_block_order(cp_size: int) -> List[int]:
    """Return the block permutation that maps zig-zag block layout to rank order.

    With ``2*cp_size`` blocks, cp-rank ``r`` owns blocks ``r`` and
    ``2*cp_size-1-r``. Concatenating each rank's ``[r, 2*cp-1-r]`` blocks in rank
    order yields the permutation ``[0, 2c-1, 1, 2c-2, ...]`` over the original
    block indices.
    """
    order = []
    for r in range(cp_size):
        order.append(r)
        order.append(2 * cp_size - 1 - r)
    return order


def zigzag_reorder(tensor: Tensor, dim: int, cp_size: int) -> Tensor:
    """Permute ``tensor`` along ``dim`` into zig-zag block order.

    ``tensor`` length along ``dim`` must be divisible by ``2 * cp_size``. After
    reordering, a contiguous split into ``cp_size`` chunks gives cp-rank ``r``
    its ``[block r, block 2*cp-1-r]`` pair.
    """
    if cp_size <= 1:
        return tensor
    n_blocks = 2 * cp_size
    seq = tensor.size(dim)
    assert seq % n_blocks == 0, f"seq len {seq} not divisible by 2*cp_size ({n_blocks})"
    blocks = list(torch.tensor_split(tensor, n_blocks, dim=dim))
    order = zigzag_block_order(cp_size)
    return torch.cat([blocks[i].contiguous() for i in order], dim=dim)


def zigzag_undo(tensor: Tensor, dim: int, cp_size: int) -> Tensor:
    """Inverse of :func:`zigzag_reorder` — restore the original block order."""
    if cp_size <= 1:
        return tensor
    n_blocks = 2 * cp_size
    seq = tensor.size(dim)
    assert seq % n_blocks == 0, f"seq len {seq} not divisible by 2*cp_size ({n_blocks})"
    blocks = list(torch.tensor_split(tensor, n_blocks, dim=dim))
    order = zigzag_block_order(cp_size)
    inverse = [0] * n_blocks
    for new_pos, orig in enumerate(order):
        inverse[orig] = new_pos
    return torch.cat([blocks[inverse[i]].contiguous() for i in range(n_blocks)], dim=dim)


#
# Packed sequences hold several documents concatenated along the sequence dim.
# For balanced causal ring attention over ``cp`` each document must be
# INDEPENDENTLY zig-zag split across the ``cp`` group. ``zigzag_reorder_varlen``
# permutes each document's tokens into zig-zag block order so a subsequent
# CONTIGUOUS split into ``cp_size`` chunks hands cp-rank ``r`` its
# ``[block r, block 2*cp-1-r]`` pair FOR EVERY DOCUMENT. Every document length
# must be divisible by ``2 * cp_size``.


def zigzag_reorder_varlen(tensor: Tensor, cu_seqlens: "torch.Tensor", dim: int, cp_size: int) -> Tensor:
    """Zig-zag reorder packed documents for a subsequent contiguous ``cp`` split.

    Each document is split into ``2*cp_size`` blocks; cp-rank ``r`` owns blocks
    ``r`` and ``2*cp_size-1-r`` of EVERY document. The output is laid out
    ``rank``-major / ``document``-minor so a plain contiguous split into
    ``cp_size`` chunks hands rank ``r`` its per-document ``[block r,
    block 2*cp-1-r]`` pairs concatenated across documents (matching
    :func:`local_cu_seqlens`). Every document length must be divisible by
    ``2 * cp_size``.
    """
    if cp_size <= 1:
        return tensor
    cu = [int(x) for x in cu_seqlens.tolist()]
    n_blocks = 2 * cp_size
    # doc_blocks[i] holds the 2*cp block slices for document i.
    doc_blocks = []
    for i in range(len(cu) - 1):
        start, end = cu[i], cu[i + 1]
        seg_len = end - start
        assert seg_len % n_blocks == 0, (
            f"document length {seg_len} not divisible by 2*cp_size ({n_blocks}); "
            "packed varlen under context-parallel requires every document length "
            "to be a multiple of 2*cp_size"
        )
        seg = tensor.narrow(dim, start, seg_len)
        doc_blocks.append(list(torch.tensor_split(seg, n_blocks, dim=dim)))
    pieces = []
    for r in range(cp_size):
        for blocks in doc_blocks:
            pieces.append(blocks[r])
            pieces.append(blocks[n_blocks - 1 - r])
    return torch.cat([p.contiguous() for p in pieces], dim=dim)


def local_cu_seqlens(cu_seqlens: "torch.Tensor", cp_size: int) -> "torch.Tensor":
    """Per-rank LOCAL cu_seqlens after per-document zig-zag slicing.

    Each document's local length is ``doc_len // cp_size`` (this rank owns 2 of
    the ``2*cp_size`` blocks). Returns a tensor of the same dtype/device.
    """
    if cp_size <= 1:
        return cu_seqlens
    cu = [int(x) for x in cu_seqlens.tolist()]
    n_blocks = 2 * cp_size
    local = [0]
    for i in range(len(cu) - 1):
        seg_len = cu[i + 1] - cu[i]
        assert seg_len % n_blocks == 0, (
            f"document length {seg_len} not divisible by 2*cp_size ({n_blocks}); "
            "packed varlen under context-parallel requires every document length "
            "to be a multiple of 2*cp_size"
        )
        local.append(local[-1] + seg_len // cp_size)
    return torch.tensor(local, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def prepare_usp_metadata(cu_seqlens: torch.Tensor, cp_size: int) -> dict:
    """Build packed Ring offsets and half indices once on the collator's CPU.

    Each document is aligned before splitting into two halves per CP rank.
    Integer gather indices avoid dynamic boolean-index sizing on accelerators.
    """
    if cu_seqlens.device.type != "cpu":
        raise ValueError("USP metadata must be precomputed on CPU before attention.")
    if cp_size < 1:
        raise ValueError("cp_size must be positive")
    boundaries = cu_seqlens.tolist()
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError("USP cu_seqlens must start at zero and contain a document")
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if any(length <= 0 or length % (2 * cp_size) for length in lengths):
        raise ValueError("USP document lengths must be positive multiples of 2 * cp_size")
    local_cu = cu_seqlens.to(torch.int32) // cp_size
    local = local_cu.tolist()
    halves = [[], []]
    for start, end in zip(local, local[1:]):
        middle = (start + end) // 2
        halves[0].extend(range(start, middle))
        halves[1].extend(range(middle, end))
    return {
        "usp_cu_seqlens": local_cu,
        "usp_max_seqlen": max(lengths) // cp_size,
        "usp_half0": torch.tensor(halves[0], dtype=torch.long),
        "usp_half1": torch.tensor(halves[1], dtype=torch.long),
    }


zigzag_reorder_packed = zigzag_reorder_varlen
