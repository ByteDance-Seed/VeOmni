# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Composable MXFP8 expert linear and recomputed SwiGLU autograd nodes.

The grouped GEMM quantization lives in mxfp8_group_gemm; the pointwise
operations reuse the existing fused Triton SwiGLU helpers.
"""

import torch
from torch.autograd.function import once_differentiable

from .mxfp8_group_gemm import mxfp8_group_gemm, mxfp8_group_wgrad
from .swiglu import apply_swiglu_clamp, merged_swiglu_backward, merged_swiglu_forward


class GroupedLinear(torch.autograd.Function):
    """BF16-facing MXFP8 grouped linear with approximate first-order gradients.

    Both backward GEMMs requantize their BF16 operands using the same K32
    recipe as forward. Each invocation owns its buffers; there is no shared
    quantized-weight cache. Save inputs only for requested weight gradients
    and weights only for requested input gradients.
    """

    @staticmethod
    def forward(ctx, x, weight, token_ends):
        """Return grouped X @ W.T and retain the operands needed by backward."""
        ctx.save_for_backward(
            x if ctx.needs_input_grad[1] else None,
            weight if ctx.needs_input_grad[0] else None,
            token_ends,
        )
        return mxfp8_group_gemm(x, weight, token_ends)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Compute requested dgrad and wgrad with independent MXFP8 quantization."""
        x, weight, token_ends = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = mxfp8_group_gemm(grad_output, weight.transpose(1, 2).contiguous(), token_ends)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = mxfp8_group_wgrad(grad_output, x, token_ends)
        return grad_x, grad_weight, None


class RecomputedSwiGLU(torch.autograd.Function):
    """Wrap existing fused SwiGLU helpers, saving the BF16 projection only."""

    @staticmethod
    def forward(ctx, projected, limit):
        """Clamp optional gate/up bounds and compute staged SiLU times up."""
        ctx.save_for_backward(projected)
        ctx.limit = limit
        gate, up, _, _ = apply_swiglu_clamp(*projected.chunk(2, dim=-1), limit)
        return merged_swiglu_forward(gate, up)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Recompute clamp masks and delegate the derivative to the shared helper."""
        (projected,) = ctx.saved_tensors
        gate, up, gate_mask, up_mask = apply_swiglu_clamp(*projected.chunk(2, dim=-1), ctx.limit)
        _, grad = merged_swiglu_backward(gate, up, grad_output, gate_mask, up_mask)
        return grad, None
