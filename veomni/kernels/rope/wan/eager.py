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

"""wan RoPE eager math (complex multiply by freqs)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ...registry import SavedState


@dataclass(frozen=True)
class _Meta:
    empty: bool
    head_dim: int


def forward(x: Tensor, freqs: Tensor, *, head_dim: int) -> tuple[Tensor, SavedState]:
    if x.numel() == 0:
        return x, SavedState((freqs,), _Meta(True, head_dim))

    shaped = x.reshape(*x.shape[:2], -1, head_dim)
    rotated = torch.view_as_complex(shaped.to(torch.float64).reshape(*shaped.shape[:3], -1, 2))
    output = torch.view_as_real(rotated * freqs).flatten(2).to(x.dtype)
    return output, SavedState((freqs,), _Meta(False, head_dim))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    (freqs,) = saved.tensors
    if meta.empty:
        return grad_output, None

    shaped = grad_output.reshape(*grad_output.shape[:2], -1, meta.head_dim)
    rotated = torch.view_as_complex(shaped.to(torch.float64).reshape(*shaped.shape[:3], -1, 2))
    dx = torch.view_as_real(rotated * freqs.conj()).flatten(2).to(grad_output.dtype)
    return dx, None
