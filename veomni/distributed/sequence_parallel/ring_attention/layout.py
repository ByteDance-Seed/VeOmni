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
