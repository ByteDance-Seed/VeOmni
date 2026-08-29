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

"""standard RMSNorm Triton impl (batch-invariant row reduction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from ...registry import SavedState
from . import eager as _eager


_kernel: Callable | None = None


def _rms_norm_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel

    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        sum_sq = tl.zeros([1], dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sum_sq += tl.sum(tl.where(mask, vals_f32 * vals_f32, 0.0))

        inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
            w = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            out = vals.to(tl.float32) * inv_rms * w.to(tl.float32)
            tl.store(output_row_start_ptr + col_idx, out.to(vals.dtype), mask=mask)

    _kernel = kernel
    return _kernel


def _rms_norm_forward(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)

    original_shape = x.shape
    weight = weight.contiguous()

    _rms_norm_kernel()[(n_rows,)](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=1024,
    )
    return output.reshape(original_shape)


@dataclass(frozen=True)
class _Meta:
    empty: bool
    eps: float


def forward(x: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
    if x.numel() == 0:
        output, saved = _eager.forward(x, weight, eps=eps)
        return output, SavedState(saved.tensors, _Meta(True, eps))

    output = _rms_norm_forward(x, weight, eps)
    return output, SavedState((x, weight), _Meta(False, eps))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    x, weight = saved.tensors
    if meta.empty:
        return _eager.backward(grad_output, SavedState((x, weight), _eager._Meta(True, meta.eps)))

    input_f32 = x.float()
    inv_rms = torch.rsqrt(input_f32.pow(2).mean(-1, keepdim=True) + meta.eps)
    normed = input_f32 * inv_rms
    grad_weight = (grad_output.float() * normed).reshape(-1, x.shape[-1]).sum(0).to(weight.dtype)
    d = grad_output.float() * weight.float()
    grad_input = (inv_rms * (d - normed * (d * normed).mean(-1, keepdim=True))).to(x.dtype)
    return grad_input, grad_weight
