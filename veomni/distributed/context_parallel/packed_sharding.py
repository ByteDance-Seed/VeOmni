# Copyright (c) 2026 ByteDance AI4SE.
# Sample-aware zigzag sharding adapted from MindSpeed get_index (BSD-3-Clause).
"""Per-sample balanced CP / hybrid sharding for packed sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .sharding import _validate_parallel_coordinate


@dataclass(frozen=True)
class PackedCPPartition:
    """Gather indices and CP-local packed metadata for one rank."""

    token_indices: Tensor  # int64 [T_local]
    local_cu_seqlens: Tensor  # int32 [num_samples + 1]
    local_max_seqlen: int
    sample_multiple: int


def _sample_lengths(cu_seqlens: Tensor) -> list[int]:
    cu = cu_seqlens.detach().cpu().tolist()
    if not cu or cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu[:3]}...")
    return [int(cu[i + 1] - cu[i]) for i in range(len(cu) - 1)]


def build_packed_cp_partition(
    cu_seqlens: Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    ulysses_size: int = 1,
    ulysses_rank: int = 0,
) -> PackedCPPartition:
    """Build sample-aware hybrid CP×Ulysses gather indices from global cu_seqlens."""
    _validate_parallel_coordinate(cp_size, cp_rank, "cp")
    _validate_parallel_coordinate(ulysses_size, ulysses_rank, "ulysses")
    multiple = 2 * cp_size * ulysses_size
    lengths = _sample_lengths(cu_seqlens)
    for idx, length in enumerate(lengths):
        if length % multiple != 0:
            raise ValueError(
                f"Packed sample {idx} length {length} must be divisible by "
                f"2 * cp_size * ulysses_size ({multiple}) for Ring CP."
            )

    points = cu_seqlens.detach().cpu().tolist()
    index_chunks: list[Tensor] = []
    local_lengths: list[int] = []
    for start, end in zip(points[:-1], points[1:]):
        start, end = int(start), int(end)
        # Per-sample zigzag CP, then contiguous Ulysses on the CP-local shard.
        size = (end - start) // (2 * cp_size)
        part1 = torch.arange(start + cp_rank * size, start + (cp_rank + 1) * size)
        part2 = torch.arange(end - (cp_rank + 1) * size, end - cp_rank * size)
        cp_local = torch.cat((part1, part2))
        if ulysses_size > 1:
            ulysses_chunk = cp_local.numel() // ulysses_size
            begin = ulysses_rank * ulysses_chunk
            cp_local = cp_local[begin : begin + ulysses_chunk]
        index_chunks.append(cp_local)
        local_lengths.append(int(cp_local.numel()))

    token_indices = torch.cat(index_chunks) if index_chunks else torch.zeros(0, dtype=torch.long)
    local_cu = torch.tensor([0] + local_lengths, dtype=torch.int32).cumsum(0).to(dtype=torch.int32)
    local_max = max(local_lengths) if local_lengths else 0
    return PackedCPPartition(
        token_indices=token_indices,
        local_cu_seqlens=local_cu,
        local_max_seqlen=local_max,
        sample_multiple=multiple,
    )


def apply_packed_cp_partition(tensor: Tensor, partition: PackedCPPartition, *, dim: int = -1) -> Tensor:
    """Gather ``tensor`` along ``dim`` using packed CP indices."""
    dim = dim % tensor.ndim
    index = partition.token_indices.to(device=tensor.device)
    if index.numel() == 0:
        sizes = list(tensor.shape)
        sizes[dim] = 0
        return tensor.new_empty(sizes)
    return tensor.index_select(dim, index).contiguous()


def pad_sample_lengths_to_multiple(lengths: list[int], multiple: int) -> list[int]:
    """Return padded lengths (each >= original, divisible by multiple)."""
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}.")
    padded = []
    for length in lengths:
        rem = length % multiple
        padded.append(length if rem == 0 else length + (multiple - rem))
    return padded


def pad_packed_tensor_to_sample_multiple(
    tensor: Tensor,
    cu_seqlens: Tensor,
    *,
    multiple: int,
    dim: int = -1,
    pad_value: int | float = 0,
) -> tuple[Tensor, Tensor]:
    """Pad each packed sample so its length is divisible by ``multiple``.

    Returns ``(padded_tensor, new_cu_seqlens)``.
    """
    dim = dim % tensor.ndim
    lengths = _sample_lengths(cu_seqlens)
    padded_lengths = pad_sample_lengths_to_multiple(lengths, multiple)
    if padded_lengths == lengths:
        return tensor, cu_seqlens.to(dtype=torch.int32)

    pieces = []
    points = cu_seqlens.detach().cpu().tolist()
    for (start, end), new_len, old_len in zip(zip(points[:-1], points[1:]), padded_lengths, lengths):
        start, end = int(start), int(end)
        piece = tensor.narrow(dim, start, end - start)
        pad_n = new_len - old_len
        if pad_n:
            pad_shape = list(piece.shape)
            pad_shape[dim] = pad_n
            piece = torch.cat(
                (piece, piece.new_full(pad_shape, fill_value=pad_value)),
                dim=dim,
            )
        pieces.append(piece)
    padded = torch.cat(pieces, dim=dim)
    new_cu = torch.tensor([0] + padded_lengths, dtype=torch.int32).cumsum(0).to(dtype=torch.int32)
    return padded.contiguous(), new_cu

def ulysses_local_cu_to_cp_local_cu(local_cu_seqlens: Tensor, ulysses_size: int) -> Tensor:
    """Scale Ulysses-local cu_seqlens to CP-local lengths after gather_seq."""
    if ulysses_size <= 1:
        return local_cu_seqlens.to(dtype=torch.int32)
    lengths = local_cu_seqlens.diff() * int(ulysses_size)
    out = torch.empty(local_cu_seqlens.numel(), dtype=torch.int32, device=local_cu_seqlens.device)
    out[0] = 0
    out[1:] = lengths.to(dtype=torch.int32).cumsum(0)
    return out


def reorder_ulysses_rank_major_to_sample_major(
    tensor: Tensor,
    ulysses_local_cu_seqlens: Tensor,
    *,
    ulysses_size: int,
    seq_dim: int,
) -> Tensor:
    """Map post-Ulysses gather layout ``[u0_packed|u1_packed|...]`` to CP-local
    sample-major ``[s0_u0|s0_u1|...|s1_u0|s1_u1|...]`` for Ring CP.
    """
    if ulysses_size <= 1:
        return tensor
    seq_dim = seq_dim % tensor.ndim
    lengths = _sample_lengths(ulysses_local_cu_seqlens)
    if not lengths:
        return tensor
    rank_span = sum(lengths)
    expected = rank_span * ulysses_size
    actual = int(tensor.size(seq_dim))
    if actual != expected:
        raise RuntimeError(
            f"Ulysses gather seq length {actual} != sum(local_cu)*ulysses_size ({expected})"
        )
    rank_blocks = tensor.split(rank_span, dim=seq_dim)
    per_sample: list[list[Tensor]] = [[] for _ in lengths]
    for block in rank_blocks:
        for i, chunk in enumerate(block.split(lengths, dim=seq_dim)):
            per_sample[i].append(chunk)
    samples = [torch.cat(chunks, dim=seq_dim) for chunks in per_sample]
    return torch.cat(samples, dim=seq_dim).contiguous()


def reorder_sample_major_to_ulysses_rank_major(
    tensor: Tensor,
    ulysses_local_cu_seqlens: Tensor,
    *,
    ulysses_size: int,
    seq_dim: int,
) -> Tensor:
    """Inverse of :func:`reorder_ulysses_rank_major_to_sample_major` before Ulysses scatter."""
    if ulysses_size <= 1:
        return tensor
    seq_dim = seq_dim % tensor.ndim
    lengths = _sample_lengths(ulysses_local_cu_seqlens)
    if not lengths:
        return tensor
    cp_lengths = [length * ulysses_size for length in lengths]
    expected = sum(cp_lengths)
    actual = int(tensor.size(seq_dim))
    if actual != expected:
        raise RuntimeError(
            f"CP-local seq length {actual} != sum(local_cu)*ulysses_size ({expected})"
        )
    samples = tensor.split(cp_lengths, dim=seq_dim)
    by_rank: list[list[Tensor]] = [[] for _ in range(ulysses_size)]
    for sample, ulen in zip(samples, lengths):
        for rank, chunk in enumerate(sample.split(ulen, dim=seq_dim)):
            by_rank[rank].append(chunk)
    ranks = [torch.cat(chunks, dim=seq_dim) for chunks in by_rank]
    return torch.cat(ranks, dim=seq_dim).contiguous()
