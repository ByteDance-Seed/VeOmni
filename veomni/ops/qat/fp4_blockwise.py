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

"""Block-wise FP4 fake quantization for quantization-aware training.

The companion to ``fp8_blockwise`` for the one place DeepSeek-V4 leaves FP8: the
routed experts of a V4-Flash checkpoint, whose ``expert_dtype`` is ``fp4``. The
geometry differs from the FP8 weight recipe and the difference matters. FP8 uses
square ``128 x 128`` tiles, one scale per tile; FP4 groups only along the input
dimension, ``32`` channels at a time, giving an ``[out, in / 32]`` scale array --
the same layout ``checkpoint_tensor_converter`` writes when it exports experts.

Scales are always ``float8_e8m0fnu``, i.e. powers of two by construction, so
unlike FP8 there is no ``scale_fmt`` to choose.

The TileLang quantizer is SM90-only and BF16-only, so both entry points here
inherit those restrictions.
"""

import torch
from torch.distributed.tensor import DTensor

from ..kernels.deepseek_v4 import fp4_act_quant


__all__ = [
    "FP4_BLOCK_SIZE",
    "fp4_fake_quant_weight",
]

# `[out, in/32]` per the checkpoint's expert scale layout.
FP4_BLOCK_SIZE = 32


class _Fp4FakeQuantWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, block_size: int) -> torch.Tensor:
        return fp4_act_quant(weight.detach(), block_size, dequant=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def fp4_fake_quant_weight(weight: torch.Tensor, block_size: int = FP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize-dequantize a weight over ``1 x block_size`` groups of input channels.

    Args:
        weight: BF16 weight whose last dimension is the input dimension and is
            divisible by ``block_size``. Leading dimensions are free: a stacked
            MoE expert weight ``[E, out, in]`` is quantized exactly as the E
            separate matrices would be, because a group never spans two rows and
            therefore never spans two experts.
        block_size: Number of input channels sharing a scale.

    Returns:
        A BF16 tensor holding the dequantized weight, differentiable into
        ``weight`` through a straight-through estimator.
    """
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"FP4 fake quantization expects a bfloat16 weight, got {weight.dtype}")
    # A DTensor reports the global shape, so the divisibility check below would
    # be answered for dimensions the local shard does not have.
    if isinstance(weight, DTensor):
        raise TypeError("FP4 fake quantization does not support a DTensor weight; pass the local shard")
    if weight.dim() < 2:
        raise ValueError(f"FP4 weight fake quantization expects at least 2 dimensions, got {tuple(weight.shape)}")
    if weight.shape[-1] % block_size:
        raise ValueError(f"weight of {weight.shape[-1]} input channels is not divisible by block_size {block_size}")
    return _Fp4FakeQuantWeight.apply(weight, block_size)
