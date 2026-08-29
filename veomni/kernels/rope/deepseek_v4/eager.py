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

from ...registry import SavedState


@dataclass(frozen=True)
class _Meta:
    empty: bool
    unsqueeze_dim: int


def _rotate_half(x: Tensor) -> Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def forward(x: Tensor, cos: Tensor, sin: Tensor, *, unsqueeze_dim: int = 1) -> tuple[Tensor, SavedState]:
    if x.numel() == 0:
        return x, SavedState((cos, sin), _Meta(True, unsqueeze_dim))

    cos_u = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin_u = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos_u.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos_u) + (_rotate_half(rope).float() * sin_u)).to(x.dtype)
    return torch.cat((nope, rotated), dim=-1), SavedState((cos, sin), _Meta(False, unsqueeze_dim))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None, None]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return grad_output, None, None

    cos, sin = saved.tensors
    cos_u = cos.repeat_interleave(2, dim=-1).unsqueeze(meta.unsqueeze_dim)
    sin_u = sin.repeat_interleave(2, dim=-1).unsqueeze(meta.unsqueeze_dim)
    rope_dim = cos_u.shape[-1]
    nope, rope = grad_output[..., :-rope_dim], grad_output[..., -rope_dim:]
    rope_f = rope.float()
    dx_rope = (rope_f * cos_u - _rotate_half(rope_f * sin_u)).to(grad_output.dtype)
    return torch.cat((nope, dx_rope), dim=-1), None, None
