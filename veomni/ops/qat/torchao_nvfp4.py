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

"""TorchAO NVFP4 fake quantization for quantization-aware training.

NVFP4 stores each group of block_size (16 in the NVIDIA FP4 format) channels
as E2M1 elements with two-level scaling: one E4M3 block scale per 16-element
group and one FP32 per-tensor scale derived from the global tensor amax.
Deployment kernels read that layout directly, so QAT trains against the rounded
values those kernels compute while the optimizer updates ordinary
high-precision parameters.

This is a pure tensor fake quantizer built on torchao's
:meth:`NVFP4Tensor.to_nvfp4` / :meth:`NVFP4Tensor.dequantize` rather than a
module-swapping wrapper. That keeps training bit-aligned with torchao's NVFP4
quantization numerics while running on any sequence length (an empty
micro-batch passes through unchanged) and on both CPU and CUDA devices,
without requiring Blackwell SM100+ native FP4 dispatch during QAT.

``torchao`` is imported lazily on the first quantizing call, so importing
``veomni.ops`` on a host without it stays free.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor


__all__ = [
    "NVFP4_BLOCK_SIZE",
    "nvfp4_fake_quant",
]

#: The NVIDIA NVFP4 group width (16 elements per E4M3 block scale).
NVFP4_BLOCK_SIZE = 16


def _nvfp4_roundtrip(x: Tensor, block_size: int) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to NVFP4 with two-level scaling."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

    # `to_nvfp4` expects a 2D or 3D contiguous tensor; flatten leading dimensions
    # into rows and restore the original shape after dequantizing.
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # Clamp per_tensor_scale away from 0 so an all-zero operand dequantizes to
    # zeros instead of dividing 0.0 / 0.0 inside torchao's nvfp4_quantize.
    per_tensor_scale = per_tensor_amax_to_scale(torch.max(torch.abs(x_2d))).clamp(min=1e-12)
    quantized = NVFP4Tensor.to_nvfp4(
        x_2d,
        block_size=block_size,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=False,
        use_triton_kernel=False,
    )
    return quantized.dequantize(x.dtype).reshape(x.shape)


class _NvFp4FakeQuant(torch.autograd.Function):
    """Straight-through NVFP4 fake quantization."""

    @staticmethod
    def forward(ctx, x: Tensor, block_size: int) -> Tensor:
        """Return the NVFP4 quantize-dequantize round trip of ``x``."""
        return _nvfp4_roundtrip(x.detach(), block_size)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Pass the output gradient straight through to the high-precision operand."""
        return grad_output, None


def nvfp4_fake_quant(x: Tensor, block_size: int = NVFP4_BLOCK_SIZE) -> Tensor:
    """Fake-quantize ``x`` in NVFP4, differentiably into ``x``.

    Args:
        x: High-precision activation or weight whose last dimension is the
            quantized dimension. Leading dimensions are free.
        block_size: Number of trailing channels sharing one E4M3 block scale
            (must be 16 for NVFP4).

    Returns:
        A tensor of the input shape and dtype holding the values an NVFP4
        deployment kernel would consume, differentiable into ``x`` through a
        straight-through estimator.

    Raises:
        TypeError: for a ``DTensor`` operand or a dtype other than BF16/FP32.
        ValueError: for a ``block_size`` other than 16, or a last dimension that
            would leave a partial group.
    """
    # A DTensor reports the global shape, so the divisibility check below would
    # validate a shape this rank does not hold, and the per-tensor ``amax`` would
    # silently depend on how the tensor is partitioned. Same contract as the FP8
    # primitives: the caller passes the local tensor.
    if isinstance(x, DTensor):
        raise TypeError("NVFP4 fake quantization does not support a DTensor operand; pass the local tensor")
    if x.numel() == 0:
        return x
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"NVFP4 fake quantization expects a bfloat16 or float32 operand, got {x.dtype}")
    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 fake quantization requires block_size={NVFP4_BLOCK_SIZE}, got {block_size}.")
    if x.shape[-1] % block_size:
        raise ValueError(
            f"NVFP4 fake quantization needs the last dimension ({x.shape[-1]}) divisible by block_size "
            f"{block_size}; the deployment kernels index one scale per group and cannot pad a partial one."
        )
    return _NvFp4FakeQuant.apply(x, block_size)
