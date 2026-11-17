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

"""TorchAO FP4 fake quantization for quantization-aware training.

Two FP4 layouts are supported, distinguished by the row's ``dtype`` argument:

``nvfp4``
    NVFP4 stores each group of 16 channels as E2M1 elements with two-level
    scaling: one E4M3 block scale per group and one FP32 per-tensor scale derived
    from the global tensor amax.

``mxfp4``
    MXFP4 stores each group of 32 channels as E2M1 elements with a single E8M0
    (power-of-two) block scale per group and no per-tensor scale.

Deployment kernels read those layouts directly, so QAT trains against the rounded
values those kernels compute while the optimizer updates ordinary high-precision
parameters.

Both are pure tensor fake quantizers built on torchao's tensor subclasses rather
than a module-swapping wrapper. That keeps training bit-aligned with torchao's
numerics while running on any sequence length (an empty micro-batch passes through
unchanged) and on both CPU and CUDA devices, without requiring Blackwell SM100+
native FP4 dispatch during QAT.

``torchao`` is imported lazily on the first quantizing call, so importing
``veomni.ops`` on a host without it stays free.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor


__all__ = [
    "NVFP4_BLOCK_SIZE",
    "SUPPORTED_DTYPES",
    "nvfp4_fake_quant",
    "torchao_fake_quant",
]

#: The NVIDIA NVFP4 group width (16 elements per E4M3 block scale).
NVFP4_BLOCK_SIZE = 16
#: The MXFP4 group width (32 elements per E8M0 block scale).
MXFP4_BLOCK_SIZE = 32

#: Formats this module can fake-quantize to, as the row's ``dtype`` values.
SUPPORTED_DTYPES = ("nvfp4", "mxfp4")

#: Each format's group width, which is the only width it defines.
_BLOCK_SIZES = {"nvfp4": NVFP4_BLOCK_SIZE, "mxfp4": MXFP4_BLOCK_SIZE}


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
    with torch.no_grad():
        quantized = _nvfp4_roundtrip(x, block_size)
    return quantized.detach() + (x - x.detach())


def _mxfp4_roundtrip(x: Tensor, block_size: int) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to MXFP4 with an E8M0 scale."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    # ``MXFP4`` is torchao's `e2m1` element type with an `e8m0` scale, one per group;
    # there is no per-tensor scale, which is the whole difference from NVFP4.
    quantized = MXTensor.to_mx(
        x_2d,
        torch.float4_e2m1fn_x2,
        block_size=block_size,
        scaling_mode="mxfp4",
        use_triton_kernel=False,
    )
    return quantized.dequantize(torch.bfloat16).reshape(x.shape)


def _check_operand(x: Tensor, dtype: str, block_size: int) -> None:
    """Validate the shared contract: local operand, supported dtype, whole groups."""
    # A DTensor reports the global shape, so the divisibility check below would
    # validate a shape this rank does not hold, and a per-tensor ``amax`` would
    # silently depend on how the tensor is partitioned. Same contract as the FP8
    # primitives: the caller passes the local tensor.
    if isinstance(x, DTensor):
        raise TypeError(f"{dtype} fake quantization does not support a DTensor operand; pass the local tensor")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{dtype} fake quantization expects a bfloat16 or float32 operand, got {x.dtype}")
    width = _BLOCK_SIZES[dtype]
    if block_size != width:
        raise ValueError(f"{dtype} fake quantization requires block_size={width}, got {block_size}.")
    if x.shape[-1] % block_size:
        raise ValueError(
            f"{dtype} fake quantization needs the last dimension ({x.shape[-1]}) divisible by block_size "
            f"{block_size}; the deployment kernels index one scale per group and cannot pad a partial one."
        )


def torchao_fake_quant(dtype: str):
    """Return ``(x, block_size=None) -> x`` fake-quantizing to ``dtype``.

    The row's ``dtype`` argument picks the format; this resolves it once, when the
    handle is built, rather than on every call. ``block_size`` defaults to the
    format's group width, which is the only width it defines.
    """
    if dtype not in _BLOCK_SIZES:
        raise ValueError(f"torchao fake quantization does not support dtype={dtype!r}; it supports {SUPPORTED_DTYPES}")

    def quantize(x: Tensor, block_size: int | None = None) -> Tensor:
        width = _BLOCK_SIZES[dtype] if block_size is None else block_size
        if x.numel() == 0:
            return x
        _check_operand(x, dtype, width)
        with torch.no_grad():
            rounded = _nvfp4_roundtrip(x, width) if dtype == "nvfp4" else _mxfp4_roundtrip(x, width)
        return rounded.detach() + (x - x.detach())

    return quantize
