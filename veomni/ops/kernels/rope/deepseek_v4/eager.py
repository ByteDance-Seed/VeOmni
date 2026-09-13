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

"""deepseek_v4 RoPE eager math (trailing interleaved slice)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """Shape metadata plus whether ``x`` was saved for table gradients."""

    empty: bool
    unsqueeze_dim: int
    table_gradients: bool


def _rotate_half(x: Tensor) -> Tensor:
    """Interleaved even/odd swap: ``(-odd, even)`` on the last dim."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _collapse_table_gradient(grad: Tensor, expanded: Tensor, table: Tensor, unsqueeze_dim: int) -> Tensor:
    """Undo broadcast, ``unsqueeze``, and last-dim ``repeat_interleave(2)``."""
    grad = grad.sum_to_size(expanded.shape).squeeze(unsqueeze_dim)
    return grad.unflatten(-1, (table.shape[-1], 2)).sum(-1).to(table.dtype)


def forward(x: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1) -> tuple[Tensor, SavedState]:
    """Rotate the trailing interleaved slice of ``x``.

    ``cos`` / ``sin`` are half-width and broadcast with ``repeat_interleave``.
    The leading ``nope`` channels pass through. ``x`` is saved only when
    ``cos`` or ``sin`` needs a gradient.
    """
    if x.numel() == 0:
        return x, SavedState((cos, sin), _Meta(True, unsqueeze_dim, False))

    cos_u = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin_u = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos_u.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos_u) + (_rotate_half(rope).float() * sin_u)).to(x.dtype)
    table_gradients = cos.requires_grad or sin.requires_grad
    tensors = (x, cos, sin) if table_gradients else (cos, sin)
    return torch.cat((nope, rotated), dim=-1), SavedState(
        tensors,
        _Meta(False, unsqueeze_dim, table_gradients),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Return gradients for ``x`` and any trainable ``cos`` / ``sin`` tables."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return grad_output, None, None

    if meta.table_gradients:
        x, cos, sin = saved.tensors
    else:
        cos, sin = saved.tensors
    cos_u = cos.repeat_interleave(2, dim=-1).unsqueeze(meta.unsqueeze_dim)
    sin_u = sin.repeat_interleave(2, dim=-1).unsqueeze(meta.unsqueeze_dim)
    rope_dim = cos_u.shape[-1]
    nope_grad, rope_grad = grad_output[..., :-rope_dim], grad_output[..., -rope_dim:]
    compute_dtype = torch.promote_types(torch.float32, torch.promote_types(cos.dtype, sin.dtype))
    rope_grad = rope_grad.to(compute_dtype)
    cos_compute = cos_u.to(compute_dtype)
    sin_compute = sin_u.to(compute_dtype)
    dx_rope = (rope_grad * cos_compute - _rotate_half(rope_grad * sin_compute)).to(grad_output.dtype)
    grad_x = torch.cat((nope_grad, dx_rope), dim=-1)

    if not meta.table_gradients:
        return grad_x, None, None

    input_rope = x[..., -rope_dim:].float().to(compute_dtype)
    grad_cos = (
        _collapse_table_gradient(rope_grad * input_rope, cos_u, cos, meta.unsqueeze_dim) if cos.requires_grad else None
    )
    grad_sin = (
        _collapse_table_gradient(rope_grad * _rotate_half(input_rope), sin_u, sin, meta.unsqueeze_dim)
        if sin.requires_grad
        else None
    )
    return grad_x, grad_cos, grad_sin
