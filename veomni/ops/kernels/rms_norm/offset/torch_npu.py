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

"""offset RMSNorm torch_npu adapter (scale is 1 + weight)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ....registry import SavedState
from ..group import apply_group_size_fallback_weighted
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Whether the eager path ran, plus ``eps`` for that fallback."""

    empty: bool
    eps: float
    use_eager: bool = False


def _eager_fallback(x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, SavedState]:
    """Run the eager pair and keep its saved tensors for backward."""
    output, saved = _eager.forward(x, weight, eps=eps)
    eager_meta = saved.metadata
    assert isinstance(eager_meta, _eager._Meta)
    return output, SavedState(saved.tensors, _Meta(eager_meta.empty, eps, use_eager=True))


def forward(x: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    """NPU fused affine RMSNorm. The fused scale is ``1 + weight``.

    Empty ``x``, and tensors that are not on NPU, fall back to the eager
    pair. The row is chosen from the host, so a CPU activation on an NPU
    machine must not enter ``npu_rms_norm``. Backward also passes
    ``1 + weight`` into ``npu_rms_norm_backward``.
    """
    scale = 1.0 + weight
    if x.numel() == 0 or x.device.type != "npu":
        return _eager_fallback(x, weight, eps)

    import torch_npu

    output, rstd = torch_npu.npu_rms_norm(x, scale, eps)
    return output, SavedState((x, weight, rstd), _Meta(False, eps))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_x, grad_weight)``. Empty inputs reuse the eager backward."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight, *optional_rstd = saved.tensors
    if meta.use_eager:
        return _eager.backward(grad_output, SavedState(saved.tensors, _eager._Meta(meta.empty, meta.eps)))

    import torch_npu

    (rstd,) = optional_rstd
    scale = 1.0 + weight
    return torch_npu.npu_rms_norm_backward(grad_output.contiguous(), x, scale, rstd)


forward, backward = apply_group_size_fallback_weighted(forward, backward, _eager.forward, _eager.backward)
