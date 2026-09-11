# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import torch
import triton
import triton.language as tl

from .triton_utils.activation import silu, silu_grad


_BLOCK_SIZE = 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _clamped_swiglu_forward_kernel(
    x,
    output,
    hidden_size,
    limit,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < hidden_size
    row_offset = row * hidden_size * 2

    gate = tl.load(x + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x + row_offset + hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)

    tl.store(output + row * hidden_size + cols, silu(gate) * up, mask=mask)


@triton.jit
def _clamped_swiglu_backward_kernel(
    x,
    grad_output,
    grad_input,
    hidden_size,
    limit,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < hidden_size
    row_offset = row * hidden_size * 2

    gate = tl.load(x + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x + row_offset + hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_output + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)

    gate_clamped = tl.minimum(gate, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)
    grad_gate = grad * up_clamped * silu_grad(gate_clamped)
    grad_gate = tl.where(gate <= limit, grad_gate, 0.0)
    grad_up = grad * silu(gate_clamped)
    grad_up = tl.where((up >= -limit) & (up <= limit), grad_up, 0.0)

    tl.store(grad_input + row_offset + cols, grad_gate, mask=mask)
    tl.store(grad_input + row_offset + hidden_size + cols, grad_up, mask=mask)


class _ClampedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float) -> torch.Tensor:
        hidden_size = x.shape[-1] // 2
        output = torch.empty((*x.shape[:-1], hidden_size), dtype=x.dtype, device=x.device)
        ctx.limit = limit
        ctx.save_for_backward(x)
        if output.numel() == 0:
            return output

        rows = x.numel() // x.shape[-1]
        grid = (rows, triton.cdiv(hidden_size, _BLOCK_SIZE))
        _clamped_swiglu_forward_kernel[grid](
            x,
            output,
            hidden_size,
            limit,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        if grad_input.numel() == 0:
            return grad_input, None

        hidden_size = x.shape[-1] // 2
        rows = x.numel() // x.shape[-1]
        grad_output = grad_output.contiguous()
        grid = (rows, triton.cdiv(hidden_size, _BLOCK_SIZE))
        _clamped_swiglu_backward_kernel[grid](
            x,
            grad_output,
            grad_input,
            hidden_size,
            ctx.limit,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return grad_input, None


def npu_triton_clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply DeepSeek-V4 clamped SwiGLU with Ascend Triton autograd."""
    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError(f"clamped SwiGLU requires an even last dimension, got shape {tuple(x.shape)}")
    if x.device.type != "npu":
        raise RuntimeError(f"Ascend Triton clamped SwiGLU requires an NPU tensor, got {x.device.type!r}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"Ascend Triton clamped SwiGLU does not support dtype {x.dtype}")
    return _ClampedSwiGLU.apply(x.contiguous(), float(limit))
