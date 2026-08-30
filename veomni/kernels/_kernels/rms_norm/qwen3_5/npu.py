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

"""qwen3_5 RMSNorm npu adapter (scale is 1 + weight)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Whether the empty-tensor path ran, plus ``eps`` for the eager fallback."""

    empty: bool
    eps: float


def forward(x: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    """NPU fused affine RMSNorm. The fused scale is ``1 + weight``.

    Empty ``x`` falls back to the eager pair. Backward also passes
    ``1 + weight`` into ``npu_rms_norm_backward``.
    """
    scale = 1.0 + weight
    if x.numel() == 0:
        output, saved = _eager.forward(x, weight, eps=eps)
        return output, SavedState(saved.tensors, _Meta(True, eps))

    import torch_npu

    output, rstd = torch_npu.npu_rms_norm(x, scale, eps)
    return output, SavedState((x, weight, rstd), _Meta(False, eps))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_x, grad_weight)``. Empty inputs reuse the eager backward."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, *optional_rstd = saved.tensors
    if meta.empty:
        return _eager.backward(grad_output, SavedState((x, weight), _eager._Meta(True, meta.eps)))

    import torch_npu

    (rstd,) = optional_rstd
    scale = 1.0 + weight
    return torch_npu.npu_rms_norm_backward(grad_output.contiguous(), x, scale, rstd)
