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

"""TorchAO MXFP4 fake quantization for quantization-aware training.

MXFP4 stores each group of ``block_size`` (32 in the OCP MX format) channels as
E2M1 elements sharing one E8M0 scale. Deployment kernels read that layout
directly, so QAT has to show training the values those kernels would compute
while the optimizer keeps updating ordinary high-precision parameters.

This is a fake quantizer, not a quantized GEMM. The round trip runs through
torchao's own :meth:`MXTensor.to_mx` / :meth:`MXTensor.dequantize`, which is what
makes training bit-aligned with torchao's ``MXFakeQuantizedLinear`` rather than
a second, drifting reimplementation of the same numerics. It is also why this
module never imports ``torchao.quantization``: the module-swapping QAT recipe
would replace layers and hide them behind a CUDA-only autograd function, while
an op-registry row only needs the tensor math.

The backward pass is the usual straight-through approximation. A
quantize-dequantize round trip is a step function whose true derivative is zero
almost everywhere; torchao's own MX QAT backward likewise multiplies the saved
dequantized operands in high precision. Both therefore propagate the same
gradients, and the only difference is that this module recomputes nothing:
the dequantized operands are already the ones the forward GEMM consumed.

``torchao`` is imported lazily on the first quantizing call, so importing
``veomni.ops`` on a host without it stays free.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate


__all__ = [
    "MXFP4_BLOCK_SIZE",
    "mxfp4_fake_quant",
    "replicated_local",
]

#: The OCP MX group width. Deployment kernels consume 32-wide groups, so a
#: different width here would silently train against a different recipe.
MXFP4_BLOCK_SIZE = 32


def _mx_roundtrip(x: Tensor, block_size: int) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to E2M1 with E8M0 scales."""
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    # `to_mx` asserts contiguity rather than normalizing, and a sliced
    # activation (head split, packed sample view) arrives non-contiguous.
    quantized = MXTensor.to_mx(
        x.contiguous(),
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.EMULATED,
    )
    return quantized.dequantize(x.dtype)


class _MxFp4FakeQuant(torch.autograd.Function):
    """Straight-through MXFP4 fake quantization."""

    @staticmethod
    def forward(ctx, x: Tensor, block_size: int) -> Tensor:
        """Return the MXFP4 quantize-dequantize round trip of ``x``."""
        return _mx_roundtrip(x.detach(), block_size)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Pass the output gradient straight through to the high-precision operand."""
        return grad_output, None


def replicated_local(tensor: Tensor, what: str) -> Tensor:
    """Return the local tensor of a replicated DTensor, or ``tensor`` unchanged.

    FSDP2 hands a projection its parameter as a ``Replicate`` DTensor after the
    all-gather, so the local shard is the whole tensor and using it is exact.
    Any other placement is refused instead of quantized: MX groups run along the
    last dimension, so a shard boundary through that dimension would pair
    channels with another shard's scale and only change the numbers.

    The bias is returned through this too. Leaving it a DTensor while the weight
    became a plain tensor makes ``addmm`` reject the call with "got mixed
    torch.Tensor and DTensor", and the bias is not quantized, so its local form
    is the same value.
    """
    if not isinstance(tensor, DTensor):
        return tensor
    if not all(isinstance(placement, Replicate) for placement in tensor.placements):
        raise TypeError(
            f"MXFP4 fake quantization requires a replicated DTensor {what}, got placements "
            f"{tuple(str(placement) for placement in tensor.placements)}; sharding the last dimension "
            "would split a quantization group across shards."
        )
    return tensor.to_local()


def mxfp4_fake_quant(x: Tensor, block_size: int = MXFP4_BLOCK_SIZE) -> Tensor:
    """Fake-quantize ``x`` in MXFP4, differentiably into ``x``.

    Args:
        x: High-precision activation or weight whose last dimension is the
            quantized dimension. Leading dimensions are free: rows are
            independent, so a stacked MoE expert weight ``[E, out, in]`` is
            quantized exactly as the ``E`` separate matrices would be.
        block_size: Number of trailing channels sharing one scale.

    Returns:
        A tensor of the input shape and dtype holding the values an MXFP4
        deployment kernel would consume, differentiable into ``x`` through a
        straight-through estimator.
    """
    if x.numel() == 0:
        return x
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"MXFP4 fake quantization expects a bfloat16 or float32 operand, got {x.dtype}")
    x = replicated_local(x, "operand")
    if x.shape[-1] % block_size:
        raise ValueError(
            f"MXFP4 fake quantization needs the last dimension ({x.shape[-1]}) divisible by block_size "
            f"{block_size}; the deployment kernels index one scale per group and cannot pad a partial one."
        )
    return _MxFp4FakeQuant.apply(x, block_size)
