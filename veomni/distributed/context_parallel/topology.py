# Copyright (c) 2026 ByteDance AI4SE.
"""Hybrid CP × Ulysses rank topology helpers.

Canonical layout (matches ``init_sequence_parallel``):
    sp_rank = cp_rank * ulysses_size + ulysses_rank

DeviceMesh dim order must therefore be ``cp`` then ``ulysses`` (CP outer,
Ulysses inner) so flattened ``sp`` ranks agree with legacy group construction.
"""

from __future__ import annotations

from typing import Sequence


def validate_hybrid_sizes(cp_size: int, ulysses_size: int) -> None:
    if cp_size < 1 or ulysses_size < 1:
        raise ValueError(f"cp_size and ulysses_size must be >= 1, got {cp_size}, {ulysses_size}.")


def sp_rank_from_coords(cp_rank: int, ulysses_rank: int, ulysses_size: int) -> int:
    validate_hybrid_sizes(1, ulysses_size)
    return cp_rank * ulysses_size + ulysses_rank


def coords_from_sp_rank(sp_rank: int, cp_size: int, ulysses_size: int) -> tuple[int, int]:
    validate_hybrid_sizes(cp_size, ulysses_size)
    if not 0 <= sp_rank < cp_size * ulysses_size:
        raise ValueError(f"sp_rank {sp_rank} out of range for CP{cp_size}×U{ulysses_size}.")
    cp_rank = sp_rank // ulysses_size
    ulysses_rank = sp_rank % ulysses_size
    return cp_rank, ulysses_rank


def cp_global_ranks_for_ulysses_rank(
    *,
    dp_index: int,
    ulysses_rank: int,
    cp_size: int,
    ulysses_size: int,
) -> list[int]:
    """Global ranks that form one CP ring (fixed DP + Ulysses coordinate)."""
    validate_hybrid_sizes(cp_size, ulysses_size)
    unified = cp_size * ulysses_size
    start = dp_index * unified + ulysses_rank
    return list(range(start, (dp_index + 1) * unified, ulysses_size))


def ulysses_global_ranks_for_cp_rank(
    *,
    dp_index: int,
    cp_rank: int,
    cp_size: int,
    ulysses_size: int,
) -> list[int]:
    """Global ranks that form one Ulysses group (fixed DP + CP coordinate)."""
    validate_hybrid_sizes(cp_size, ulysses_size)
    unified = cp_size * ulysses_size
    start = dp_index * unified + cp_rank * ulysses_size
    return list(range(start, start + ulysses_size))


def ordered_cp_global_ranks(group_ranks: Sequence[int]) -> list[int]:
    """Return CP ring ranks sorted by CP rank (ascending)."""
    return sorted(group_ranks)
