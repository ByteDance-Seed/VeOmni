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

"""interleave RoPE eager math (interleaved pairs in, rotate-half out)."""

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


def _half_tables(cos: Tensor, sin: Tensor, unsqueeze_dim: int) -> tuple[Tensor, Tensor]:
    """Take the first half of ``cat(freqs, freqs)`` tables and broadcast."""
    half = cos.shape[-1] // 2
    return cos[..., :half].unsqueeze(unsqueeze_dim), sin[..., :half].unsqueeze(unsqueeze_dim)


def _apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate interleaved pairs and emit rotate-half layout."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)


def _grad_x(grad_output: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Inverse of ``_apply`` back onto the interleaved last dim."""
    grad_even, grad_odd = grad_output.chunk(2, dim=-1)
    dx_even = grad_even * cos + grad_odd * sin
    dx_odd = grad_odd * cos - grad_even * sin
    return torch.stack((dx_even, dx_odd), dim=-1).flatten(-2)


def _collapse_table_gradient(grad: Tensor, expanded: Tensor, table: Tensor, unsqueeze_dim: int) -> Tensor:
    """Undo broadcasting and restore the unused second half of ``cat(freqs, freqs)``."""
    half = table.shape[-1] // 2
    collapsed = grad.sum_to_size(expanded.shape).squeeze(unsqueeze_dim).to(table.dtype)
    full = table.new_zeros(table.shape)
    full[..., :half] = collapsed
    return full


def forward(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Rotate interleaved ``q`` / ``k`` pairs and return rotate-half layout.

    ``cos`` / ``sin`` are the HF full-width ``cat(freqs, freqs)`` tables. Only
    the first half is used. ``position_ids`` is retained for compatibility and
    does not participate in the math. Empty inputs are returned unchanged.
    """
    del position_ids
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True, unsqueeze_dim, False))

    cos_u, sin_u = _half_tables(cos, sin, unsqueeze_dim)
    table_gradients = cos.requires_grad or sin.requires_grad
    tensors = (q, k, cos, sin) if table_gradients else (cos, sin)
    return (
        _apply(q, cos_u, sin_u),
        _apply(k, cos_u, sin_u),
    ), SavedState(tensors, _Meta(False, unsqueeze_dim, table_gradients))


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
    cos_u, sin_u = _half_tables(cos, sin, meta.unsqueeze_dim)
    dq = _grad_x(grad_q, cos_u, sin_u)
    dk = _grad_x(grad_k, cos_u, sin_u)
    if not meta.table_gradients:
        return dq, dk, None, None, None, None

    grad_q_even, grad_q_odd = grad_q.chunk(2, dim=-1)
    grad_k_even, grad_k_odd = grad_k.chunk(2, dim=-1)
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    grad_cos = None
    if cos.requires_grad:
        grad_cos = _collapse_table_gradient(
            grad_q_even * q_even + grad_q_odd * q_odd, cos_u, cos, meta.unsqueeze_dim
        ) + _collapse_table_gradient(grad_k_even * k_even + grad_k_odd * k_odd, cos_u, cos, meta.unsqueeze_dim)
    grad_sin = None
    if sin.requires_grad:
        grad_sin = _collapse_table_gradient(
            grad_q_odd * q_even - grad_q_even * q_odd, sin_u, sin, meta.unsqueeze_dim
        ) + _collapse_table_gradient(grad_k_odd * k_even - grad_k_even * k_odd, sin_u, sin, meta.unsqueeze_dim)
    return dq, dk, grad_cos, grad_sin, None, None
