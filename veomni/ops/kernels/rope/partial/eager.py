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

"""partial RoPE eager math (rotate a prefix of each head)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """Broadcast metadata and whether input tensors were saved for table gradients."""

    empty: bool
    unsqueeze_dim: int
    table_gradients: bool


def _rotate_half(x: Tensor) -> Tensor:
    """Swap the two halves of the last dim, negating the second."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_prefix(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x[..., :cos.shape[-1]]``. The remaining channels pass through."""
    rotary_dim = cos.shape[-1]
    rotated, passed = x[..., :rotary_dim], x[..., rotary_dim:]
    embedded = (rotated * cos) + (_rotate_half(rotated) * sin)
    return torch.cat((embedded, passed), dim=-1)


def _grad_prefix(grad_output: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Inverse-rotate the prefix. The unrotated suffix is copied through."""
    rotary_dim = cos.shape[-1]
    rotated, passed = grad_output[..., :rotary_dim], grad_output[..., rotary_dim:]
    grad_rotated = (rotated * cos) - _rotate_half(rotated * sin)
    return torch.cat((grad_rotated, passed), dim=-1)


def _collapse_table_gradient(grad: Tensor, expanded: Tensor, table: Tensor, unsqueeze_dim: int) -> Tensor:
    """Undo broadcasting and the table's inserted head dimension."""
    return grad.sum_to_size(expanded.shape).squeeze(unsqueeze_dim).to(table.dtype)


def forward(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Rotate a prefix of each head. The remaining channels pass through.

    Prefix width is ``cos.shape[-1]``. Empty inputs are returned unchanged.
    """
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True, unsqueeze_dim, False))

    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    table_gradients = cos.requires_grad or sin.requires_grad
    tensors = (q, k, cos, sin) if table_gradients else (cos, sin)
    return (
        _apply_prefix(q, cos_u, sin_u),
        _apply_prefix(k, cos_u, sin_u),
    ), SavedState(tensors, _Meta(False, unsqueeze_dim, table_gradients))


def backward(
    grad_output: tuple[Tensor, Tensor], saved: SavedState
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, None]:
    """Return q/k, optional table, and ``unsqueeze_dim`` gradients."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_q, grad_k = grad_output
    if meta.empty:
        return grad_q, grad_k, None, None, None

    if meta.table_gradients:
        q, k, cos, sin = saved.tensors
    else:
        cos, sin = saved.tensors
    cos_u = cos.unsqueeze(meta.unsqueeze_dim)
    sin_u = sin.unsqueeze(meta.unsqueeze_dim)
    dq = _grad_prefix(grad_q, cos_u, sin_u)
    dk = _grad_prefix(grad_k, cos_u, sin_u)
    if not meta.table_gradients:
        return dq, dk, None, None, None

    rotary_dim = cos.shape[-1]
    grad_q_rot, q_rot = grad_q[..., :rotary_dim], q[..., :rotary_dim]
    grad_k_rot, k_rot = grad_k[..., :rotary_dim], k[..., :rotary_dim]
    grad_cos = None
    if cos.requires_grad:
        grad_cos = _collapse_table_gradient(
            grad_q_rot * q_rot, cos_u, cos, meta.unsqueeze_dim
        ) + _collapse_table_gradient(grad_k_rot * k_rot, cos_u, cos, meta.unsqueeze_dim)
    grad_sin = None
    if sin.requires_grad:
        grad_sin = _collapse_table_gradient(
            grad_q_rot * _rotate_half(q_rot), sin_u, sin, meta.unsqueeze_dim
        ) + _collapse_table_gradient(grad_k_rot * _rotate_half(k_rot), sin_u, sin, meta.unsqueeze_dim)
    return dq, dk, grad_cos, grad_sin, None
