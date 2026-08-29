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

"""standard RMSNorm eager math (offset 0, llama-style cast)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ...registry import SavedState


@dataclass(frozen=True)
class _Meta:
    empty: bool
    eps: float


def _batch_dims(tensor: Tensor) -> tuple[int, ...]:
    return tuple(range(tensor.ndim - 1))


def forward(x: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    if x.numel() == 0:
        return x * weight, SavedState((x, weight), _Meta(True, eps))

    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    output = weight * (x_f * rstd).to(x.dtype)
    return output, SavedState((x, weight, rstd), _Meta(False, eps))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, *optional_rstd = saved.tensors
    if meta.empty:
        return torch.zeros_like(x), torch.zeros_like(weight)

    (rstd,) = optional_rstd
    x_f = x.float()
    n = x.shape[-1]
    scaled_grad = (grad_output * weight).float()
    grad_weight = (grad_output * (x_f * rstd).to(x.dtype)).sum(dim=_batch_dims(x))
    grad_x = rstd * scaled_grad - (rstd.pow(3) / n) * x_f * (scaled_grad * x_f).sum(dim=-1, keepdim=True)
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)
