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

"""Shared ``normalized_shape`` coercion for affine LayerNorm rows."""

from __future__ import annotations

import numbers
from typing import Any

from torch import Tensor


def coerce_normalized_shape(
    normalized_shape: int | tuple[int, ...] | None,
    weight: Tensor,
) -> tuple[int, ...]:
    """Return a concrete last-dim shape for affine LayerNorm.

    ``None`` uses ``weight.shape``. An integer becomes a one-element tuple.
    """
    if normalized_shape is None:
        return tuple(weight.shape)
    if isinstance(normalized_shape, numbers.Integral):
        return (int(normalized_shape),)
    return tuple(int(size) for size in normalized_shape)


def empty_affine_output(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply affine scale and shift without a reduction, for empty ``x``."""
    return x * weight + bias


def as_size(normalized_shape: tuple[int, ...]) -> Any:
    """Return ``torch.Size`` for vendor kernels that reject a plain tuple."""
    import torch

    return torch.Size(normalized_shape)
