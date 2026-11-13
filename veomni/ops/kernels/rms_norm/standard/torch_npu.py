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

"""standard RMSNorm torch_npu adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from ..group import apply_group_size_fallback_weighted
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Whether the empty-tensor path ran, plus ``eps`` for the eager fallback."""

    empty: bool
    eps: float
    x_dtype: torch.dtype | None = None


def forward(x: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    """NPU fused affine RMSNorm (offset 0).

    Empty ``x`` falls back to the eager pair. Otherwise saves ``(x, weight, rstd)``.
    """
    if x.numel() == 0:
        output, saved = _eager.forward(x, weight, eps=eps)
        return output, SavedState(saved.tensors, _Meta(True, eps))

    import torch_npu

    # aclnnRmsNorm has no mixed-dtype overload (fp32 x with bf16 gamma or vice
    # versa); align defensively and cast the gradient back in ``backward``.
    x_dtype = x.dtype if x.dtype != weight.dtype else None
    if x_dtype is not None:
        x = x.to(weight.dtype)
    output, rstd = torch_npu.npu_rms_norm(x, weight, eps)
    return output, SavedState((x, weight, rstd), _Meta(False, eps, x_dtype))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_x, grad_weight)``. Empty inputs reuse the eager backward."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, *optional_rstd = saved.tensors
    if meta.empty:
        return _eager.backward(grad_output, SavedState((x, weight), _eager._Meta(True, meta.eps)))

    import torch_npu

    (rstd,) = optional_rstd
    grad_x, grad_weight = torch_npu.npu_rms_norm_backward(grad_output.contiguous(), x, weight, rstd)
    if meta.x_dtype is not None:
        grad_x = grad_x.to(meta.x_dtype)
    return grad_x, grad_weight


forward, backward = apply_group_size_fallback_weighted(forward, backward, _eager.forward, _eager.backward)
