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
    """NPU fused affine RMSNorm (offset 0) under a same-dtype contract.

    Non-empty ``x`` and ``weight`` must share one dtype: ``aclnnRmsNorm`` has
    no mixed-dtype overload, so misaligned inputs are rejected instead of
    being silently cast (callers align on their side). Empty ``x``, and
    tensors that are not on NPU, fall back to the eager pair. The row is
    chosen from the host, so a CPU activation on an NPU machine must not
    enter ``npu_rms_norm``. Otherwise saves ``(x, weight, rstd)``.
    """
    if x.numel() == 0:
        return _eager_fallback(x, weight, eps)
    if x.dtype != weight.dtype:
        raise TypeError(
            "rms_norm/torch_npu requires x and weight to share one dtype "
            f"(got x={x.dtype}, weight={weight.dtype}); cast on the caller side."
        )
    if x.device.type != "npu":
        return _eager_fallback(x, weight, eps)

    import torch_npu

    output, rstd = torch_npu.npu_rms_norm(x, weight, eps)
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
    return torch_npu.npu_rms_norm_backward(grad_output.contiguous(), x, weight, rstd)


forward, backward = apply_group_size_fallback_weighted(forward, backward, _eager.forward, _eager.backward)
