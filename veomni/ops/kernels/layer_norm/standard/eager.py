# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""standard affine LayerNorm eager math via ``native_layer_norm``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from .shape import coerce_normalized_shape, empty_affine_output


@dataclass(frozen=True)
class _Meta:
    """Last-dim shape, ``eps``, and whether the empty-tensor path ran."""

    normalized_shape: tuple[int, ...]
    eps: float
    empty: bool


def forward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    normalized_shape: int | tuple[int, ...] | None = None,
    eps: float,
) -> tuple[Tensor, SavedState]:
    """Affine LayerNorm matching ``F.layer_norm``.

    Empty ``x`` skips the reduction and returns ``x * weight + bias``. Saves
    ``(x, weight, bias, mean, rstd)`` when the reduction ran.
    """
    shape = coerce_normalized_shape(normalized_shape, weight)
    if x.numel() == 0:
        return empty_affine_output(x, weight, bias), SavedState((x, weight, bias), _Meta(shape, eps, True))

    output, mean, rstd = torch.native_layer_norm(x, shape, weight, bias, eps)
    return output, SavedState((x, weight, bias, mean, rstd), _Meta(shape, eps, False))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(grad_x, grad_weight, grad_bias)`` matching the positional tensors."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, bias, *optional_stats = saved.tensors
    if meta.empty:
        return torch.zeros_like(x), torch.zeros_like(weight), torch.zeros_like(bias)

    mean, rstd = optional_stats
    grad_x, grad_weight, grad_bias = torch.ops.aten.native_layer_norm_backward(
        grad_output,
        x,
        meta.normalized_shape,
        mean,
        rstd,
        weight,
        bias,
        [True, True, True],
    )
    return grad_x, grad_weight, grad_bias
