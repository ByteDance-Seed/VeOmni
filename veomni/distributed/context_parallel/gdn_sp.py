# Copyright (c) 2025 VeOmni Authors.
# Load-balancing helpers adapted from MindSpeed-LLM gdn_context_parallel (BSD-3-Clause).
"""GDN sequence-parallel helpers for hybrid Ring-CP layouts."""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from ..parallel_state import get_parallel_state
from .sharding import balanced_cp_restore, balanced_cp_to_rank_major


def resolve_gdn_sp_group() -> tuple[Optional[ProcessGroup], int, int, bool]:
    """Return (group, sp_size, head_rank, need_zigzag_restore) for GDN A2A.

    - Ulysses-only: use ulysses group (no zigzag restore).
    - CP enabled (with or without Ulysses): use unified SP group and restore
      canonical order after gather / re-apply zigzag before scatter.
    """
    state = get_parallel_state()
    if state.cp_enabled:
        group = state.sp_group
        if group is None and not state.ulysses_enabled:
            group = state.cp_group
        if group is None:
            raise RuntimeError("CP-enabled GDN requires unified sequence-parallel group.")
        sp_rank = state.sp_rank if state.sp_rank >= 0 else (state.cp_rank if state.cp_rank >= 0 else 0)
        return group, state.sp_size, sp_rank, True
    if state.ulysses_enabled:
        return state.ulysses_group, state.ulysses_size, state.ulysses_rank, False
    return None, 1, 0, False


def maybe_restore_canonical(tensor: Tensor, *, enabled: bool, cp_size: int, dim: int = 1) -> Tensor:
    if not enabled or cp_size <= 1:
        return tensor
    return balanced_cp_restore(tensor, cp_size=cp_size, dim=dim)


def maybe_to_rank_major(tensor: Tensor, *, enabled: bool, cp_size: int, dim: int = 1) -> Tensor:
    if not enabled or cp_size <= 1:
        return tensor
    return balanced_cp_to_rank_major(tensor, cp_size=cp_size, dim=dim)


def gdn_sp_world_size(group: Optional[ProcessGroup]) -> int:
    if group is None:
        return 1
    return dist.get_world_size(group)
