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

"""vision RoPE eager math ([S, H, D] query/key, unsqueeze heads)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """Whether the empty path ran and inputs were saved for table gradients."""

    empty: bool
    table_gradients: bool


def _rotate_half(x: Tensor) -> Tensor:
    """Swap the two halves of the last dim, negating the second."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotate-half RoPE: ``x * cos + rotate_half(x) * sin``."""
    return (x * cos) + (_rotate_half(x) * sin)


def _grad_x(grad_output: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Inverse rotate-half: ``g * cos - rotate_half(g * sin)``."""
    return (grad_output * cos) - _rotate_half(grad_output * sin)


def _collapse_table_gradient(grad: Tensor, expanded: Tensor, table: Tensor) -> Tensor:
    """Undo broadcasting and the fixed vision head dimension."""
    return grad.sum_to_size(expanded.shape).squeeze(-2).to(table.dtype)


def forward(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Rotate every channel of ``[S, H, D]`` query/key.

    ``position_ids`` and ``unsqueeze_dim`` retain the previous public call
    face. Vision tensors always insert the broadcast dimension at ``-2``.
    Empty inputs are returned unchanged.
    """
    del position_ids, unsqueeze_dim
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True, False))

    q_f, k_f = q.float(), k.float()
    cos_u, sin_u = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = _apply(q_f, cos_u, sin_u).to(q.dtype)
    k_embed = _apply(k_f, cos_u, sin_u).to(k.dtype)
    table_gradients = cos.requires_grad or sin.requires_grad
    tensors = (q, k, cos, sin) if table_gradients else (cos, sin)
    return (q_embed, k_embed), SavedState(tensors, _Meta(False, table_gradients))


def backward(
    grad_output: tuple[Tensor, Tensor], saved: SavedState
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, None, None]:
    """Return q/k, optional table, and compatibility-argument gradients."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_q, grad_k = grad_output
    if meta.empty:
        return grad_q, grad_k, None, None, None, None

    if meta.table_gradients:
        q, k, cos, sin = saved.tensors
    else:
        cos, sin = saved.tensors
    cos_u, sin_u = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    dq = _grad_x(grad_q.float(), cos_u, sin_u).to(grad_q.dtype)
    dk = _grad_x(grad_k.float(), cos_u, sin_u).to(grad_k.dtype)
    if not meta.table_gradients:
        return dq, dk, None, None, None, None

    grad_q_f, grad_k_f = grad_q.float(), grad_k.float()
    q_f, k_f = q.float(), k.float()
    grad_cos = None
    if cos.requires_grad:
        grad_cos = _collapse_table_gradient(grad_q_f * q_f, cos_u, cos) + _collapse_table_gradient(
            grad_k_f * k_f, cos_u, cos
        )
    grad_sin = None
    if sin.requires_grad:
        grad_sin = _collapse_table_gradient(grad_q_f * _rotate_half(q_f), sin_u, sin) + _collapse_table_gradient(
            grad_k_f * _rotate_half(k_f), sin_u, sin
        )
    return dq, dk, grad_cos, grad_sin, None, None
