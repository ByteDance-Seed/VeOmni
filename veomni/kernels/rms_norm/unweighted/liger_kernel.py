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

"""unweighted RMSNorm Liger adapter (no affine weight, llama casting)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from veomni.kernels.registry import SavedState

from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    empty: bool
    casting_mode: int
    block_size: int
    num_warps: int
    eps: float


def forward(x: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    if x.numel() == 0:
        output, saved = _eager.forward(x, eps=eps)
        return output, SavedState(saved.tensors, _Meta(True, 0, 0, 0, eps))

    from liger_kernel.ops.rms_norm import rms_norm_forward

    output, saved_x, rstd, block_size, num_warps, casting_mode = rms_norm_forward(
        x.contiguous(),
        None,
        eps,
        0.0,
        "llama",
        None,
    )
    return output, SavedState(
        (saved_x, rstd),
        _Meta(False, casting_mode, block_size, num_warps, eps),
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, *optional_rstd = saved.tensors
    if meta.empty:
        return _eager.backward(grad_output, SavedState((x,), _eager._Meta(True, meta.eps)))

    from liger_kernel.ops.rms_norm import rms_norm_backward

    (rstd,) = optional_rstd
    grad_x, grad_weight = rms_norm_backward(
        grad_output.contiguous(),
        x,
        None,
        rstd,
        0.0,
        meta.casting_mode,
        meta.block_size,
        meta.num_warps,
        False,
        None,
    )
    if grad_weight is not None:
        raise RuntimeError("unweighted Liger backward unexpectedly returned a weight grad")
    return (grad_x,)
