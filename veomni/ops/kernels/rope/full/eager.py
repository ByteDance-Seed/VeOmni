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

"""full RoPE eager math (rotate every channel)."""

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
    compute_in_fp32: bool


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


def _collapse_table_gradient(grad: Tensor, expanded: Tensor, table: Tensor, unsqueeze_dim: int) -> Tensor:
    """Undo broadcasting and the table's inserted head dimension."""
    return grad.sum_to_size(expanded.shape).squeeze(unsqueeze_dim).to(table.dtype)


def _is_vision_layout(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> bool:
    """Whether tensors use the HF vision ``[S, H, D]`` / ``[S, D]`` layout."""
    return q.ndim == 3 and k.ndim == 3 and cos.ndim == 2 and sin.ndim == 2


def forward(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Rotate every channel of ``q`` and ``k`` by ``cos`` / ``sin``.

    ``position_ids`` is retained for compatibility; the supplied tables are
    already position-selected, so it does not participate in the math.
    ``unsqueeze_dim`` broadcasts the tables onto the head axis. Rank-3
    ``[S, H, D]`` inputs with rank-2 tables use the HF vision behavior:
    the head axis is inferred and the rotation is evaluated in FP32 before
    casting back. Empty inputs are returned unchanged.
    """
    del position_ids
    vision_layout = _is_vision_layout(q, k, cos, sin)
    broadcast_dim = -2 if vision_layout else unsqueeze_dim
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True, broadcast_dim, False, vision_layout))

    q_compute = q.float() if vision_layout else q
    k_compute = k.float() if vision_layout else k
    cos_u = cos.unsqueeze(broadcast_dim)
    sin_u = sin.unsqueeze(broadcast_dim)
    if vision_layout:
        cos_u = cos_u.float()
        sin_u = sin_u.float()
    table_gradients = cos.requires_grad or sin.requires_grad
    tensors = (q, k, cos, sin) if table_gradients else (cos, sin)
    q_embed = _apply(q_compute, cos_u, sin_u).to(q.dtype) if vision_layout else _apply(q_compute, cos_u, sin_u)
    k_embed = _apply(k_compute, cos_u, sin_u).to(k.dtype) if vision_layout else _apply(k_compute, cos_u, sin_u)
    return (q_embed, k_embed), SavedState(tensors, _Meta(False, broadcast_dim, table_gradients, vision_layout))


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
    cos_u = cos.unsqueeze(meta.unsqueeze_dim)
    sin_u = sin.unsqueeze(meta.unsqueeze_dim)
    grad_q_compute = grad_q.float() if meta.compute_in_fp32 else grad_q
    grad_k_compute = grad_k.float() if meta.compute_in_fp32 else grad_k
    if meta.compute_in_fp32:
        cos_u = cos_u.float()
        sin_u = sin_u.float()
    dq = _grad_x(grad_q_compute, cos_u, sin_u)
    dk = _grad_x(grad_k_compute, cos_u, sin_u)
    if meta.compute_in_fp32:
        dq = dq.to(grad_q.dtype)
        dk = dk.to(grad_k.dtype)
    if not meta.table_gradients:
        return dq, dk, None, None, None, None

    q_compute = q.float() if meta.compute_in_fp32 else q
    k_compute = k.float() if meta.compute_in_fp32 else k
    grad_cos = None
    if cos.requires_grad:
        grad_cos = _collapse_table_gradient(
            grad_q_compute * q_compute, cos_u, cos, meta.unsqueeze_dim
        ) + _collapse_table_gradient(grad_k_compute * k_compute, cos_u, cos, meta.unsqueeze_dim)
    grad_sin = None
    if sin.requires_grad:
        grad_sin = _collapse_table_gradient(
            grad_q_compute * _rotate_half(q_compute), sin_u, sin, meta.unsqueeze_dim
        ) + _collapse_table_gradient(grad_k_compute * _rotate_half(k_compute), sin_u, sin, meta.unsqueeze_dim)
    return dq, dk, grad_cos, grad_sin, None, None
