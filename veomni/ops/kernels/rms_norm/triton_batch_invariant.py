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

H100 optimization (single-pass): when the hidden dimension fits in one block
(``n_cols <= MAX_FUSED_COLS``), the row is loaded from HBM **once** into
registers, the inverse RMS is computed, and the same cached values are
normalized in place. This halves input HBM reads (3x -> 2x total traffic on
the dominant tensors) and removes the Python-level reduction loop, which is the
dominant cost in the short-sequence (seqlen 100-200) regime where the kernel is
launch/occupancy bound. The reduction stays within a single program (one row
per program), so results remain batch-invariant / deterministic. For hidden
dims larger than ``MAX_FUSED_COLS`` the original two-pass loop is used.
"""

import torch
import triton
import triton.language as tl


# Largest hidden dim handled by the single-pass path. Covers all common LLM
# hidden sizes (<= 16384). Larger dims fall back to the streaming two-pass loop.
MAX_FUSED_COLS = 16384


@triton.jit
def _rms_norm_kernel_single_pass(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass RMSNorm: load the row once, reduce, normalize from registers."""
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols

    # Single HBM read of the row, kept in registers for the whole kernel.
    vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
    vals_f32 = vals.to(tl.float32)

    sum_sq = tl.sum(vals_f32 * vals_f32, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)

    w = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
    out = vals_f32 * inv_rms * w.to(tl.float32)
    tl.store(output_row_start_ptr + col_idx, out.to(vals.dtype), mask=mask)


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
    """Batch-invariant RMS normalization: each row processed independently.

    Streaming two-pass fallback for hidden dims that do not fit in one block.
    """
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


def _num_warps_for(block_size: int) -> int:
    # Tuned on H100 (seqlen 100-200 grid). num_warps depends ONLY on block_size
    # (== next_pow2(hidden)) so results stay batch-invariant across row counts.
    if block_size <= 2048:
        return 4
    if block_size <= 4096:
        return 8
    return 16


def _rms_norm_forward(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    if n_cols <= MAX_FUSED_COLS:
        block_size = triton.next_power_of_2(n_cols)
        _rms_norm_kernel_single_pass[(n_rows,)](
            input_2d,
            weight,
            output,
            input_2d.stride(0),
            output.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=_num_warps_for(block_size),
        )
    else:
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


@triton.jit
def _rms_norm_bwd_kernel(
    input_ptr,
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    prod_ptr,  # [N, n_cols] fp32: grad_output * normed (reduced over rows for grad_weight)
    input_row_stride,
    grad_output_row_stride,
    grad_input_row_stride,
    prod_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass fused RMSNorm backward (grad_input + per-row grad_weight product).

    Each program owns one row, so the per-row reductions use a fixed reduction
    tree => batch-invariant / deterministic, matching the forward kernel. The
    cross-row reduction for grad_weight is left to a deterministic ``sum(0)`` on
    ``prod`` by the caller (identical reduction domain to the original torch
    backward).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    x_ptr = input_ptr + row_idx * input_row_stride
    go_ptr = grad_output_ptr + row_idx * grad_output_row_stride

    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols

    x = tl.load(x_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(go_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)

    # inv_rms recomputed from x (same as torch backward: variance = mean(x^2)).
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    normed = x * inv_rms

    # grad_weight contribution for this row (write out for deterministic sum(0)).
    prod = go * normed
    tl.store(prod_ptr + row_idx * prod_row_stride + col_idx, prod, mask=mask)

    # grad_input = inv_rms * (d - normed * mean(d * normed)),  d = grad_output * weight
    d = go * w
    mean_dn = tl.sum(d * normed, axis=0) / n_cols
    grad_input = inv_rms * (d - normed * mean_dn)
    tl.store(
        grad_input_ptr + row_idx * grad_input_row_stride + col_idx,
        grad_input.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _rms_norm_backward(input, weight, grad_output, eps):
    """Fused backward when hidden fits one block; torch fallback otherwise."""
    orig_shape = input.shape
    x2d = input.reshape(-1, input.shape[-1])
    go2d = grad_output.reshape(-1, input.shape[-1]).contiguous()
    n_rows, n_cols = x2d.shape

    grad_input = torch.empty_like(x2d)
    prod = torch.empty(n_rows, n_cols, device=x2d.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(n_cols)
    _rms_norm_bwd_kernel[(n_rows,)](
        x2d,
        go2d,
        weight.contiguous(),
        grad_input,
        prod,
        x2d.stride(0),
        go2d.stride(0),
        grad_input.stride(0),
        prod.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps_for(block_size),
    )
    grad_weight = prod.sum(0).to(weight.dtype)
    return grad_input.reshape(orig_shape), grad_weight


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
        if input.shape[-1] <= MAX_FUSED_COLS:
            grad_input, grad_weight = _rms_norm_backward(input, weight, grad_output, eps)
            return grad_input, grad_weight, None
        # Streaming fallback (hidden dim larger than one block): original torch path.
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
