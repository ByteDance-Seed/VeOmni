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

"""Fused LayerNorm forward used by async Ulysses QKV.

LayerNorm is not a registry row yet. Compound QKV calls this CUDA extension
directly so it can save ``mean`` / ``invvar`` and pair them with
``backward.layer_norm_backward``. RMSNorm goes through a nested kernel instead.
"""

from __future__ import annotations

import importlib
import numbers
from typing import Any

from torch import Tensor


_fused_layer_norm_cuda = None


def normalize_shape(normalized_shape: int | tuple[int, ...] | None) -> Any:
    """Coerce ``normalized_shape`` to ``torch.Size`` for the fused LayerNorm kernel."""
    if normalized_shape is None:
        return None
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    import torch

    return torch.Size(normalized_shape)


def layernorm_forward(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor,
    normalized_shape: Any,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(output, mean, invvar)`` from ``fused_layer_norm_cuda.forward_affine``."""
    global _fused_layer_norm_cuda
    if _fused_layer_norm_cuda is None:
        _fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return _fused_layer_norm_cuda.forward_affine(hidden, normalized_shape, weight, bias, eps)
