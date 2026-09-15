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

"""DeepSeek-V4 RMSNorm eager math (offset 0, fp32 affine scale)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """Whether the empty-tensor path ran, plus ``eps`` for the eager fallback."""

    empty: bool
    eps: float


def forward(x: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    """Affine RMSNorm with offset 0 and an fp32 weight multiply.

    DeepSeek-V4 normalizes ``x`` in fp32, multiplies the normalized value by
    ``weight.float()``, then casts the result back to ``x.dtype``. Casting the
    normalized value before the affine multiply changes BF16 rounding.
    """
    if x.numel() == 0:
        return (x.float() * weight.float()).to(x.dtype), SavedState((x, weight), _Meta(True, eps))

    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    output = (weight.float() * (x_f * rstd)).to(x.dtype)
    return output, SavedState((x, weight, rstd), _Meta(False, eps))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_x, grad_weight)`` using the fp32 affine contract."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, *optional_rstd = saved.tensors
    if meta.empty:
        return torch.zeros_like(x), torch.zeros_like(weight)

    (rstd,) = optional_rstd
    x_f = x.float()
    normalized = x_f * rstd
    scaled_grad = grad_output.float() * weight.float()
    grad_weight = (grad_output.float() * normalized).sum_to_size(weight.shape)
    grad_x = rstd * scaled_grad - (rstd.pow(3) / x.shape[-1]) * x_f * (scaled_grad * x_f).sum(dim=-1, keepdim=True)
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)
