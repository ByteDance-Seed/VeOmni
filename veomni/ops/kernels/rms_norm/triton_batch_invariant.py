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

"""Batch-invariant (deterministic) RMSNorm Triton kernel.

Originally adapted from https://github.com/thinking-machines-lab/batch_invariant_ops.
Used by the DeepSeek V3 deterministic RMSNorm path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Batch-invariant RMS normalization: each row processed independently."""
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


def _rms_norm_forward(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    _rms_norm_kernel[(n_rows,)](
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


class BatchInvariantRMSNormFunction(torch.autograd.Function):
    """Batch-invariant RMSNorm with autograd support for training."""

    @staticmethod
    def forward(ctx, input, weight, eps):
        output = _rms_norm_forward(input, weight, eps)
        ctx.save_for_backward(input, weight, output)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, output = ctx.saved_tensors
        eps = ctx.eps
        input_f32 = input.float()
        variance = input_f32.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        normed = input_f32 * inv_rms
        grad_weight = (grad_output.float() * normed).reshape(-1, input.shape[-1]).sum(0).to(weight.dtype)
        grad_out_f32 = grad_output.float()
        weight_f32 = weight.float()
        d = grad_out_f32 * weight_f32
        grad_input = (inv_rms * (d - normed * (d * normed).mean(-1, keepdim=True))).to(input.dtype)
        return grad_input, grad_weight, None


def batch_invariant_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in replacement for RMSNorm.forward with batch-invariant Triton kernel."""
    return BatchInvariantRMSNormFunction.apply(input, weight, eps)


__all__ = [
    "BatchInvariantRMSNormFunction",
    "batch_invariant_rms_norm",
]
