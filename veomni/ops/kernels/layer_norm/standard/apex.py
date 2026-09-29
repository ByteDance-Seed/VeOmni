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

"""standard affine LayerNorm via Apex ``fused_layer_norm_cuda``."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

import torch
from torch import Tensor

from ....registry import SavedState
from .shape import as_size, coerce_normalized_shape, empty_affine_output


@dataclass(frozen=True)
class _Meta:
    """Last-dim shape, ``eps``, and whether the empty-tensor path ran."""

    normalized_shape: tuple[int, ...]
    eps: float
    empty: bool


_fused_layer_norm_cuda: ModuleType | None = None


def _get_fused_layer_norm_cuda() -> ModuleType:
    """Lazy-import the Apex fused LayerNorm CUDA extension."""
    global _fused_layer_norm_cuda
    if _fused_layer_norm_cuda is None:
        _fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return _fused_layer_norm_cuda


def forward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    normalized_shape: int | tuple[int, ...] | None = None,
    eps: float,
) -> tuple[Tensor, SavedState]:
    """Affine LayerNorm via ``fused_layer_norm_cuda.forward_affine``.

    Empty ``x`` skips the extension and returns ``x * weight + bias``. Saves
    ``(x, weight, bias, mean, invvar)`` when the reduction ran.
    """
    shape = coerce_normalized_shape(normalized_shape, weight)
    if x.numel() == 0:
        return empty_affine_output(x, weight, bias), SavedState((x, weight, bias), _Meta(shape, eps, True))

    output, mean, invvar = _get_fused_layer_norm_cuda().forward_affine(x, as_size(shape), weight, bias, eps)
    return output, SavedState((x, weight, bias, mean, invvar), _Meta(shape, eps, False))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(grad_x, grad_weight, grad_bias)`` from ``backward_affine``."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, bias, *optional_stats = saved.tensors
    if meta.empty:
        return torch.zeros_like(x), torch.zeros_like(weight), torch.zeros_like(bias)

    mean, invvar = optional_stats
    return _get_fused_layer_norm_cuda().backward_affine(
        grad_output.contiguous(),
        mean,
        invvar,
        x,
        as_size(meta.normalized_shape),
        weight,
        bias,
        meta.eps,
        False,
    )
