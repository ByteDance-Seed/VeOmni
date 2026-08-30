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
    """Empty flag plus the ``unsqueeze_dim`` used to broadcast ``cos`` / ``sin``."""

    empty: bool
    unsqueeze_dim: int


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


def forward(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1
) -> tuple[tuple[Tensor, Tensor], SavedState]:
    """Rotate a prefix of each head. The remaining channels pass through.

    Prefix width is ``cos.shape[-1]``. Empty inputs are returned unchanged.
    Backward returns ``(dq, dk, None, None)``.
    """
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True, unsqueeze_dim))

    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    return (
        _apply_prefix(q, cos_u, sin_u),
        _apply_prefix(k, cos_u, sin_u),
    ), SavedState((cos, sin), _Meta(False, unsqueeze_dim))


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor, Tensor, None, None]:
    """Return ``(dq, dk, None, None)``. ``cos`` / ``sin`` are not differentiated."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_q, grad_k = grad_output
    if meta.empty:
        return grad_q, grad_k, None, None

    cos, sin = saved.tensors
    cos_u = cos.unsqueeze(meta.unsqueeze_dim)
    sin_u = sin.unsqueeze(meta.unsqueeze_dim)
    return _grad_prefix(grad_q, cos_u, sin_u), _grad_prefix(grad_k, cos_u, sin_u), None, None
