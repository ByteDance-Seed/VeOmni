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

"""Sample-aware physical sharding for packed CP×Ulysses sequences."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor


def _parallel_coordinate(size: int, rank: int, name: str) -> tuple[int, int]:
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError(f"{name}_size must be an integer")
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise TypeError(f"{name}_rank must be an integer")
    size, rank = int(size), int(rank)
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid {name} coordinate rank={rank}, size={size}")
    return size, rank


def _cu_points(cu_seqlens: Tensor | Sequence[int]) -> list[int]:
    if isinstance(cu_seqlens, Tensor):
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens must be one-dimensional, got {tuple(cu_seqlens.shape)}")
        raw_points = cu_seqlens.detach().cpu().tolist()
    else:
        raw_points = list(cu_seqlens)
    points: list[int] = []
    for index, point in enumerate(raw_points):
        if isinstance(point, bool) or not isinstance(point, Integral):
            raise TypeError(f"cu_seqlens[{index}] must be an integer")
        points.append(int(point))
    if not points or points[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(end < start for start, end in zip(points, points[1:])):
        raise ValueError("cu_seqlens must be monotonic")
    return points


def _sample_lengths(cu_seqlens: Tensor | Sequence[int]) -> list[int]:
    points = _cu_points(cu_seqlens)
    lengths = [int(end) - int(start) for start, end in zip(points, points[1:])]
    return lengths


@dataclass(frozen=True)
class PackedContextParallelPartition:
    """Gather indices and local packed metadata for one CP×Ulysses rank."""

    token_indices: Tensor
    local_cu_seqlens: Tensor
    local_max_seqlen: int
    sample_multiple: int


def build_packed_context_parallel_partition(
    cu_seqlens: Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    ulysses_size: int = 1,
    ulysses_rank: int = 0,
) -> PackedContextParallelPartition:
    """Build per-sample zigzag CP indices followed by contiguous Ulysses slicing."""
    cp_size, cp_rank = _parallel_coordinate(cp_size, cp_rank, "cp")
    ulysses_size, ulysses_rank = _parallel_coordinate(ulysses_size, ulysses_rank, "ulysses")
    sample_multiple = 2 * cp_size * ulysses_size
    lengths = _sample_lengths(cu_seqlens)
    for sample_index, length in enumerate(lengths):
        if length % sample_multiple:
            raise ValueError(
                f"packed sample {sample_index} length {length} must be divisible by "
                f"2 * cp_size * ulysses_size ({sample_multiple})"
            )

    points = [int(point) for point in cu_seqlens.detach().cpu().tolist()]
    index_chunks: list[Tensor] = []
    local_lengths: list[int] = []
    for start, end in zip(points, points[1:]):
        half = (end - start) // (2 * cp_size)
        first = torch.arange(start + cp_rank * half, start + (cp_rank + 1) * half)
        second = torch.arange(end - (cp_rank + 1) * half, end - cp_rank * half)
        cp_local = torch.cat((first, second))
        ulysses_chunk = cp_local.numel() // ulysses_size
        cp_local = cp_local.narrow(0, ulysses_rank * ulysses_chunk, ulysses_chunk)
        index_chunks.append(cp_local)
        local_lengths.append(int(cp_local.numel()))

    token_indices = torch.cat(index_chunks) if index_chunks else torch.empty(0, dtype=torch.long)
    local_cu = torch.tensor([0] + local_lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
    return PackedContextParallelPartition(
        token_indices=token_indices,
        local_cu_seqlens=local_cu,
        local_max_seqlen=max(local_lengths, default=0),
        sample_multiple=sample_multiple,
    )


def apply_packed_context_parallel_partition(
    tensor: Tensor,
    partition: PackedContextParallelPartition,
    *,
    dim: int = -1,
) -> Tensor:
    """Gather a packed tensor along ``dim`` with a physical CP partition."""
    dim %= tensor.ndim
    index = partition.token_indices.to(device=tensor.device)
    if index.numel() == 0:
        shape = list(tensor.shape)
        shape[dim] = 0
        return tensor.new_empty(shape)
    return tensor.index_select(dim, index).contiguous()


def padded_sample_lengths(lengths: Sequence[int], multiple: int) -> list[int]:
    if isinstance(multiple, bool) or not isinstance(multiple, Integral):
        raise TypeError("multiple must be an integer")
    multiple = int(multiple)
    if multiple < 1:
        raise ValueError("multiple must be positive")
    output: list[int] = []
    for index, length in enumerate(lengths):
        if isinstance(length, bool) or not isinstance(length, Integral):
            raise TypeError(f"lengths[{index}] must be an integer")
        length = int(length)
        if length < 0:
            raise ValueError(f"lengths[{index}] must be non-negative")
        output.append(((length + multiple - 1) // multiple) * multiple)
    return output


def pad_packed_samples(
    tensor: Tensor,
    cu_seqlens: Tensor,
    *,
    multiple: int,
    dim: int = -1,
    pad_value: int | float = 0,
) -> tuple[Tensor, Tensor]:
    """Pad every packed sample independently to ``multiple`` tokens."""
    dim %= tensor.ndim
    lengths = _sample_lengths(cu_seqlens)
    padded_lengths = padded_sample_lengths(lengths, multiple)
    if sum(lengths) != int(tensor.size(dim)):
        raise ValueError("packed tensor length does not match cu_seqlens")
    if padded_lengths == lengths:
        return tensor, cu_seqlens.to(dtype=torch.int32)

    pieces: list[Tensor] = []
    points = [int(point) for point in cu_seqlens.detach().cpu().tolist()]
    for start, end, padded_length in zip(points, points[1:], padded_lengths):
        piece = tensor.narrow(dim, start, end - start)
        pad_length = padded_length - (end - start)
        if pad_length:
            pad_shape = list(piece.shape)
            pad_shape[dim] = pad_length
            piece = torch.cat((piece, piece.new_full(pad_shape, pad_value)), dim=dim)
        pieces.append(piece)
    padded = torch.cat(pieces, dim=dim) if pieces else tensor
    padded_cu = torch.tensor([0] + padded_lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
    return padded.contiguous(), padded_cu


def ulysses_local_cu_from_global(
    global_cu_seqlens: Tensor,
    *,
    cp_size: int,
    ulysses_size: int,
) -> Tensor:
    """Derive one CP×Ulysses rank's per-sample CU from global padded CU."""
    cp_size, _ = _parallel_coordinate(cp_size, 0, "cp")
    ulysses_size, _ = _parallel_coordinate(ulysses_size, 0, "ulysses")
    lengths = _sample_lengths(global_cu_seqlens)
    divisor = cp_size * ulysses_size
    if any(length % divisor for length in lengths):
        raise ValueError(f"global packed lengths must be divisible by cp_size * ulysses_size ({divisor})")
    local_lengths = [length // divisor for length in lengths]
    return global_cu_seqlens.new_tensor([0] + local_lengths, dtype=torch.int32).cumsum(0).to(torch.int32)


def ulysses_local_head_count(total_heads: int, ulysses_size: int) -> int:
    """Return the head shard width; context parallelism never shards heads."""
    if isinstance(total_heads, bool) or not isinstance(total_heads, Integral):
        raise TypeError("total_heads must be an integer")
    ulysses_size, _ = _parallel_coordinate(ulysses_size, 0, "ulysses")
    total_heads = int(total_heads)
    if total_heads < 1 or total_heads % ulysses_size:
        raise ValueError(f"total_heads ({total_heads}) must be divisible by ulysses_size ({ulysses_size})")
    return total_heads // ulysses_size


def reorder_ulysses_rank_major_to_sample_major(
    tensor: Tensor,
    ulysses_local_cu_seqlens: Tensor | Sequence[int],
    *,
    ulysses_size: int,
    sequence_dim: int,
) -> Tensor:
    """Interleave Ulysses rank blocks per packed sample after sequence gather."""
    if ulysses_size <= 1:
        return tensor
    sequence_dim %= tensor.ndim
    lengths = _sample_lengths(ulysses_local_cu_seqlens)
    if not lengths:
        return tensor
    rank_span = sum(lengths)
    if int(tensor.size(sequence_dim)) != rank_span * ulysses_size:
        raise ValueError("Ulysses gathered sequence length does not match packed metadata")
    rank_blocks = tensor.split(rank_span, dim=sequence_dim)
    per_sample: list[list[Tensor]] = [[] for _ in lengths]
    for block in rank_blocks:
        for sample_index, chunk in enumerate(block.split(lengths, dim=sequence_dim)):
            per_sample[sample_index].append(chunk)
    return torch.cat([torch.cat(chunks, dim=sequence_dim) for chunks in per_sample], dim=sequence_dim).contiguous()


def reorder_sample_major_to_ulysses_rank_major(
    tensor: Tensor,
    ulysses_local_cu_seqlens: Tensor | Sequence[int],
    *,
    ulysses_size: int,
    sequence_dim: int,
) -> Tensor:
    """Invert :func:`reorder_ulysses_rank_major_to_sample_major`."""
    if ulysses_size <= 1:
        return tensor
    sequence_dim %= tensor.ndim
    lengths = _sample_lengths(ulysses_local_cu_seqlens)
    if not lengths:
        return tensor
    sample_lengths = [length * ulysses_size for length in lengths]
    if int(tensor.size(sequence_dim)) != sum(sample_lengths):
        raise ValueError("sample-major sequence length does not match packed metadata")
    samples = tensor.split(sample_lengths, dim=sequence_dim)
    per_rank: list[list[Tensor]] = [[] for _ in range(ulysses_size)]
    for sample, rank_length in zip(samples, lengths):
        for rank, chunk in enumerate(sample.split(rank_length, dim=sequence_dim)):
            per_rank[rank].append(chunk)
    return torch.cat([torch.cat(chunks, dim=sequence_dim) for chunks in per_rank], dim=sequence_dim).contiguous()


__all__ = [
    "PackedContextParallelPartition",
    "apply_packed_context_parallel_partition",
    "build_packed_context_parallel_partition",
    "pad_packed_samples",
    "padded_sample_lengths",
    "reorder_sample_major_to_ulysses_rank_major",
    "reorder_ulysses_rank_major_to_sample_major",
    "ulysses_local_cu_from_global",
    "ulysses_local_head_count",
]
