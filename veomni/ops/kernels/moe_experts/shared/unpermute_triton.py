# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Triton EP combine kernels preserving PyTorch multiply/cast backward semantics."""

import torch
import triton
import triton.language as tl

from veomni.utils.device import get_torch_device


@triton.jit
def _combine(X, W, Index, Y, H: tl.constexpr, PRODUCT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    w = tl.load(W + row).to(tl.float32)
    token = tl.load(Index + row).to(tl.int64)
    value = (x * w).to(PRODUCT).to(tl.float32)
    tl.atomic_add(Y + token * H + col, value, col < H, sem="relaxed")


@triton.jit
def _combine_backward(
    X,
    W,
    Index,
    DY,
    DX,
    DW,
    H: tl.constexpr,
    PRODUCT: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    token = tl.load(Index + row).to(tl.int64)
    grad = tl.load(DY + token * H + col, col < H, other=0).to(tl.float32)
    if NEED_X:
        w = tl.load(W + row).to(tl.float32)
        tl.store(DX + row * H + col, (grad * w).to(PRODUCT), col < H)
    if NEED_W:
        x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
        # Autograd rounds each product before sum_to_size for low-precision
        # multiplication, then casts the reduction to the weight dtype.
        dw = tl.sum((grad * x).to(PRODUCT).to(tl.float32), 0)
        tl.store(DW + row, dw)


class WeightedUnpermute(torch.autograd.Function):
    """Combine valid repeated token indices; differentiate tokens and weights.

    Each index must lie in ``[0, num_tokens)``. The EP permutation produces
    this mapping; range validation on the host would synchronize the device.
    Like scatter-add, floating-point atomic accumulation is nondeterministic."""

    @staticmethod
    def forward(ctx, tokens, weights, indices, num_tokens):
        """Use one small FP32 output accumulator instead of routed FP32 copies."""
        if tokens.ndim != 2 or weights.shape != tokens.shape[:1] or indices.shape != weights.shape:
            raise ValueError("moe_unpermute expects [R,H] tokens and [R] weights/indices.")
        if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32) or weights.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("moe_unpermute supports FP16, BF16 and FP32 values.")
        if indices.dtype not in (torch.int32, torch.int64) or num_tokens < 0:
            raise ValueError("moe_unpermute requires integer indices and a nonnegative token count.")
        if num_tokens == 0 and tokens.shape[0]:
            raise ValueError("moe_unpermute cannot route rows into an empty output.")
        if weights.device != tokens.device or indices.device != tokens.device:
            raise ValueError("moe_unpermute inputs must share a device.")
        tokens, weights, indices = tokens.contiguous(), weights.contiguous(), indices.contiguous()
        dtype = torch.promote_types(tokens.dtype, weights.dtype)
        product = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype]
        output = torch.zeros((num_tokens, tokens.shape[1]), device=tokens.device, dtype=torch.float32)
        ctx.save_for_backward(tokens, weights, indices)
        ctx.product = product
        if tokens.numel():
            with get_torch_device().device(tokens.device):
                _combine[(tokens.shape[0], triton.cdiv(tokens.shape[1], 256))](
                    tokens, weights, indices, output, tokens.shape[1], product, 256, enable_fp_fusion=False
                )
        return output.to(dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Gather output gradients and reduce router gradients without expanded temporaries."""
        tokens, weights, indices = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dx = torch.empty_like(tokens) if ctx.needs_input_grad[0] else None
        dw = torch.empty_like(weights) if ctx.needs_input_grad[1] else None
        if tokens.numel() and (dx is not None or dw is not None):
            with get_torch_device().device(tokens.device):
                _combine_backward[(tokens.shape[0],)](
                    tokens,
                    weights,
                    indices,
                    grad_output,
                    dx,
                    dw,
                    tokens.shape[1],
                    ctx.product,
                    dx is not None,
                    dw is not None,
                    triton.next_power_of_2(tokens.shape[1]),
                    num_warps=8,
                    enable_fp_fusion=False,
                )
        elif dw is not None:
            dw.zero_()
        return dx, dw, None, None
