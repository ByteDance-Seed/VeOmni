# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

"""Keep empty packed segments out of native Ascend GDN recurrences.

The vendored NPU kernels assume every varlen sequence owns at least one token.
Lossless context parallelism can legitimately produce rank-local layouts such
as ``[0, 0, 32, 64]``: the first global sample is empty on this rank while the
remaining samples are active.  Passing that repeated CU point to the native
recurrence makes its tensor/list chunk ordinals disagree and can hang backward.

This module changes only the sequence *ordinal* seen by the native recurrence.
Token tensors stay untouched.  The public wrappers compact ``cu_seqlens`` and
``initial_state`` before their custom autograd Function, then restore a full-N
final state afterwards.  Keeping those transforms outside the custom Function
lets ordinary ``index_select`` / ``index_copy`` autograd restore full-N state
gradients without handwritten scatter logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActiveVarlenSegments:
    """Immutable mapping between full packed segments and active segments."""

    cu_seqlens: Tensor
    cu_seqlens_list: Sequence[int]
    active_indices: tuple[int, ...]
    compact_cu_seqlens: Tensor
    compact_cu_seqlens_list: Sequence[int]
    active_index_tensor: Tensor | None

    @property
    def sequence_count(self) -> int:
        return len(self.cu_seqlens_list) - 1

    @property
    def active_count(self) -> int:
        return len(self.active_indices)

    @property
    def all_active(self) -> bool:
        return self.active_count == self.sequence_count

    @property
    def all_empty(self) -> bool:
        return self.active_count == 0

    @property
    def has_empty(self) -> bool:
        return not self.all_active

    def compact_initial_state(self, initial_state: Tensor | None) -> Tensor | None:
        """Select active state rows; preserve object identity on the fast path."""
        if initial_state is None or self.all_active:
            return initial_state
        if self.all_empty:
            return initial_state[:0]
        assert self.active_index_tensor is not None
        active_index_tensor = self.active_index_tensor.to(device=initial_state.device)
        return initial_state.index_select(0, active_index_tensor)

    def restore_final_state(
        self,
        compact_final_state: Tensor | None,
        initial_state: Tensor | None,
    ) -> Tensor | None:
        """Restore native active-only final state to the original full-N layout."""
        if compact_final_state is None or self.all_active:
            return compact_final_state

        if initial_state is None:
            shape = (self.sequence_count, *compact_final_state.shape[1:])
            base = compact_final_state.new_zeros(shape)
        else:
            base = initial_state.to(device=compact_final_state.device, dtype=compact_final_state.dtype)

        if self.all_empty:
            return base
        assert self.active_index_tensor is not None
        active_index_tensor = self.active_index_tensor.to(device=compact_final_state.device)
        return torch.index_copy(base, 0, active_index_tensor, compact_final_state)


def _validated_host_points(
    cu_seqlens: Tensor,
    cu_seqlens_list: Sequence[int] | None,
    *,
    token_count: int,
) -> Sequence[int]:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
        raise ValueError("cu_seqlens must be a non-empty one-dimensional tensor")

    if cu_seqlens_list is None:
        points: Sequence[int] = tuple(int(point) for point in cu_seqlens.detach().cpu().tolist())
    else:
        points = cu_seqlens_list

    if len(points) != int(cu_seqlens.numel()):
        raise ValueError(
            "cu_seqlens and cu_seqlens_list must contain the same number of points, "
            f"got {int(cu_seqlens.numel())} and {len(points)}"
        )

    normalized: list[int] = []
    for point in points:
        if isinstance(point, bool) or not isinstance(point, Integral):
            raise TypeError("cu_seqlens_list points must be integers")
        normalized.append(int(point))
    if not normalized or normalized[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(right < left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("cu_seqlens must be monotonically non-decreasing")
    if normalized[-1] != int(token_count):
        raise ValueError(f"cu_seqlens must end at the token count {int(token_count)}, got {normalized[-1]}")

    # CPU inputs are cheap to compare exactly.  Device callers pass the host
    # list produced by the same precompute contract; forcing a per-layer D2H
    # synchronization merely to repeat that comparison would regress training.
    if cu_seqlens.device.type == "cpu" and normalized != [int(point) for point in cu_seqlens.tolist()]:
        raise ValueError("cu_seqlens_list does not match cu_seqlens")
    return points if all(int(point) == normalized[index] for index, point in enumerate(points)) else tuple(normalized)


def build_active_varlen_segments(
    cu_seqlens: Tensor,
    *,
    cu_seqlens_list: Sequence[int] | None,
    token_count: int,
    initial_state: Tensor | None,
) -> ActiveVarlenSegments:
    """Compile active packed segments without moving or padding token tensors."""
    points = _validated_host_points(cu_seqlens, cu_seqlens_list, token_count=token_count)
    sequence_count = len(points) - 1
    if initial_state is not None and int(initial_state.shape[0]) != sequence_count:
        raise ValueError(
            "The number of initial states must match the number of input sequences, "
            f"got {int(initial_state.shape[0])} and {sequence_count}."
        )

    active_indices = tuple(
        index for index, (left, right) in enumerate(zip(points, points[1:])) if int(right) > int(left)
    )
    if len(active_indices) == sequence_count:
        return ActiveVarlenSegments(
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=points,
            active_indices=active_indices,
            compact_cu_seqlens=cu_seqlens,
            compact_cu_seqlens_list=points,
            active_index_tensor=None,
        )

    compact_points = [0]
    compact_points.extend(int(points[index + 1]) for index in active_indices)
    boundary_indices = [0]
    boundary_indices.extend(index + 1 for index in active_indices)
    boundary_index_tensor = torch.tensor(boundary_indices, dtype=torch.long, device=cu_seqlens.device)
    compact_cu_seqlens = cu_seqlens.index_select(0, boundary_index_tensor)
    active_index_tensor = torch.tensor(active_indices, dtype=torch.long, device=cu_seqlens.device)
    return ActiveVarlenSegments(
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=points,
        active_indices=active_indices,
        compact_cu_seqlens=compact_cu_seqlens,
        compact_cu_seqlens_list=compact_points,
        active_index_tensor=active_index_tensor,
    )


def empty_varlen_result(
    output: Tensor,
    *,
    dependencies: Sequence[Tensor],
    plan: ActiveVarlenSegments,
    initial_state: Tensor | None,
    output_final_state: bool,
    state_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor | None]:
    """Return the mathematically empty recurrence with live zero-valued edges."""
    dependency = output.new_zeros(())
    for tensor in dependencies:
        edge = tensor.sum() if tensor.numel() == 0 else tensor[(0,) * tensor.ndim]
        dependency = dependency + edge.to(dtype=dependency.dtype) * 0
    output = output + dependency
    if not output_final_state:
        return output, None
    if initial_state is not None:
        return output, initial_state + dependency.to(dtype=initial_state.dtype)
    final_state = output.new_zeros((plan.sequence_count, *state_shape), dtype=torch.float32)
    return output, final_state + dependency.to(dtype=final_state.dtype)


__all__ = ["ActiveVarlenSegments", "build_active_varlen_segments", "empty_varlen_result"]
