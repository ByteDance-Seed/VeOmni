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

from ...registry import SavedState


@dataclass(frozen=True)
class _Meta:
    empty: bool


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _grad_x(grad_output: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (grad_output * cos) - _rotate_half(grad_output * sin)


def forward(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[tuple[Tensor, Tensor], SavedState]:
    if q.numel() == 0 or k.numel() == 0:
        return (q, k), SavedState((cos, sin), _Meta(True))

    q_f, k_f = q.float(), k.float()
    cos_u, sin_u = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = _apply(q_f, cos_u, sin_u).to(q.dtype)
    k_embed = _apply(k_f, cos_u, sin_u).to(k.dtype)
    return (q_embed, k_embed), SavedState((cos, sin), _Meta(False))


def backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor, Tensor, None, None]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    grad_q, grad_k = grad_output
    if meta.empty:
        return grad_q, grad_k, None, None

    cos, sin = saved.tensors
    cos_u, sin_u = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    dq = _grad_x(grad_q.float(), cos_u, sin_u).to(grad_q.dtype)
    dk = _grad_x(grad_k.float(), cos_u, sin_u).to(grad_k.dtype)
    return dq, dk, None, None
