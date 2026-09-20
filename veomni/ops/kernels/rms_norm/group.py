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

"""Optional last-dim grouping for RMSNorm variants.

Each kernel module rebinds its exported ``forward`` / ``backward`` through
these helpers so callers always see a ``group_size`` keyword. ``None``
forwards to the raw pair unchanged, including ``SavedState``.

Eager pairs reshape the last dim and keep the original math. Fused pairs
that require a 1D weight delegate grouped calls to the eager pair.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from ...registry import SavedState


@dataclass(frozen=True)
class GroupedState:
    """Outer metadata when ``group_size`` reshaped the last dim."""

    x_shape: tuple[int, ...]
    weight_shape: tuple[int, ...] | None
    inner: Any


def group_last_dim(tensor: Tensor, group_size: int) -> Tensor:
    """Tile ``tensor``'s last dim into ``(..., n_groups, group_size)``."""
    last = tensor.shape[-1]
    if last % group_size != 0:
        raise ValueError(f"last dim ({last}) must be divisible by group_size ({group_size})")
    return tensor.reshape(*tensor.shape[:-1], last // group_size, group_size)


def apply_group_size_weighted(
    raw_forward: Callable[..., tuple[Tensor, SavedState]],
    raw_backward: Callable[..., tuple[Tensor, Tensor]],
) -> tuple[Callable[..., tuple[Tensor, SavedState]], Callable[..., tuple[Tensor, Tensor]]]:
    """Add optional ``group_size`` around a weighted RMSNorm pair.

    ``group_size is None`` calls the raw pair unchanged, including SavedState.
    """

    def forward(x: Tensor, weight: Tensor, *, eps: float, group_size: int | None = None) -> tuple[Tensor, SavedState]:
        if group_size is None:
            return raw_forward(x, weight, eps=eps)
        output, saved = raw_forward(group_last_dim(x, group_size), group_last_dim(weight, group_size), eps=eps)
        return output.reshape(x.shape), SavedState(
            saved.tensors, GroupedState(tuple(x.shape), tuple(weight.shape), saved.metadata)
        )

    def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
        meta = saved.metadata
        if not isinstance(meta, GroupedState):
            return raw_backward(grad_output, saved)
        grad_x, grad_weight = raw_backward(
            grad_output.reshape(saved.tensors[0].shape), SavedState(saved.tensors, meta.inner)
        )
        return grad_x.reshape(meta.x_shape), grad_weight.reshape(meta.weight_shape)

    return forward, backward


def apply_group_size_fallback_weighted(
    raw_forward: Callable[..., tuple[Tensor, SavedState]],
    raw_backward: Callable[..., tuple[Tensor, Tensor]],
    fallback_forward: Callable[..., tuple[Tensor, SavedState]],
    fallback_backward: Callable[..., tuple[Tensor, Tensor]],
) -> tuple[Callable[..., tuple[Tensor, SavedState]], Callable[..., tuple[Tensor, Tensor]]]:
    """Expose ``group_size`` on a fused pair that cannot reshape its weight.

    ``group_size is None`` keeps the fused pair. Otherwise the already
    group-aware eager pair runs.
    """

    def forward(x: Tensor, weight: Tensor, *, eps: float, group_size: int | None = None) -> tuple[Tensor, SavedState]:
        if group_size is None:
            return raw_forward(x, weight, eps=eps)
        return fallback_forward(x, weight, eps=eps, group_size=group_size)

    def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
        if isinstance(saved.metadata, GroupedState):
            return fallback_backward(grad_output, saved)
        return raw_backward(grad_output, saved)

    return forward, backward


def apply_group_size_unweighted(
    raw_forward: Callable[..., tuple[Tensor, SavedState]],
    raw_backward: Callable[..., tuple[Tensor]],
) -> tuple[Callable[..., tuple[Tensor, SavedState]], Callable[..., tuple[Tensor]]]:
    """Add optional ``group_size`` around an unweighted RMSNorm pair."""

    def forward(x: Tensor, *, eps: float, group_size: int | None = None) -> tuple[Tensor, SavedState]:
        if group_size is None:
            return raw_forward(x, eps=eps)
        output, saved = raw_forward(group_last_dim(x, group_size), eps=eps)
        return output.reshape(x.shape), SavedState(saved.tensors, GroupedState(tuple(x.shape), None, saved.metadata))

    def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
        meta = saved.metadata
        if not isinstance(meta, GroupedState):
            return raw_backward(grad_output, saved)
        (grad_x,) = raw_backward(grad_output.reshape(saved.tensors[0].shape), SavedState(saved.tensors, meta.inner))
        return (grad_x.reshape(meta.x_shape),)

    return forward, backward
