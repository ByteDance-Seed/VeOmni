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

"""Bit-exact Torch references for the TileLang FP8 and FP4 quantizers."""

from __future__ import annotations

import torch
from torch import Tensor


FP8_MAX = 448.0
FP8_MAX_INV = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
FP4_MAX = 6.0
FP4_MAX_INV = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32)
FP4_POSITIVE_VALUES = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32)


def reference_scale(amax: Tensor, scale_fmt: str | None) -> Tensor:
    """Reproduce the kernel's FP32 reciprocal multiply and optional E8M0 rounding."""
    scaled = amax * FP8_MAX_INV.to(amax.device)
    if scale_fmt is None:
        return scaled
    bits = scaled.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    log2_ceil = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def _reference_fp4_scale(amax: Tensor) -> Tensor:
    """Round ``amax / 6`` upward to the E8M0 power-of-two scale."""
    scaled = amax * FP4_MAX_INV.to(amax.device)
    bits = scaled.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    log2_ceil = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def _encode_fp4_e2m1(values: Tensor) -> Tensor:
    """Encode logical values into E2M1 nibbles with round-to-nearest-even ties."""
    magnitudes = values.abs().unsqueeze(-1)
    levels = FP4_POSITIVE_VALUES.to(values.device)
    distances = (magnitudes - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    even_codes = (torch.arange(levels.numel(), device=values.device) % 2 == 0).view(*((1,) * values.ndim), -1)
    codes = torch.where(candidates & even_codes, levels.new_tensor(1), levels.new_tensor(0)).argmax(dim=-1)
    no_even_tie = ~(candidates & even_codes).any(dim=-1)
    codes = torch.where(no_even_tie, distances.argmin(dim=-1), codes).to(torch.uint8)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def reference_fp4_act_quant(
    x: Tensor,
    block_size: int = 32,
    dequant: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Reproduce row-wise MXFP4 E2M1 quantization and packed-nibble layout.

    Algorithm source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py
    """
    features = x.shape[-1]
    assert features % block_size == 0 and features % 2 == 0
    blocks = x.float().reshape(-1, features // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * (2**-126))
    scales = _reference_fp4_scale(amax)
    normalized = (blocks / scales).clamp(-FP4_MAX, FP4_MAX)
    codes = _encode_fp4_e2m1(normalized)
    logical = FP4_POSITIVE_VALUES.to(x.device)[(codes & 0x7).long()]
    logical = torch.where((codes & 0x8) != 0, -logical, logical)
    if dequant:
        return (logical * scales).reshape(x.shape).to(x.dtype)

    flat_codes = codes.reshape(*x.shape[:-1], features)
    packed = flat_codes[..., 0::2] | (flat_codes[..., 1::2] << 4)
    quantized = packed.contiguous().view(torch.float4_e2m1fn_x2)
    return quantized, scales.squeeze(-1).reshape(*x.shape[:-1], features // block_size).to(torch.float8_e8m0fnu)


def reference_act_quant(
    x: Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    dequant: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Reproduce row-wise block FP8 quantization operation for operation."""
    features = x.shape[-1]
    assert features % block_size == 0
    blocks = x.float().reshape(-1, features // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scales = reference_scale(amax, scale_fmt)
    quantized = (blocks / scales).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if dequant:
        return (quantized.float() * scales).reshape(x.shape).to(x.dtype)
    return (
        quantized.reshape(x.shape),
        scales.reshape(*x.shape[:-1], features // block_size).to(scale_dtype),
    )


def reference_fp8_weight_quant(
    x: Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    dequant: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Reproduce square-tile FP8 weight quantization operation for operation."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    rows, cols = x.shape
    tiles = x.float().contiguous().view(rows // block_size, block_size, cols // block_size, block_size)
    amax = tiles.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scales = reference_scale(amax, scale_fmt)
    quantized = (tiles / scales[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if dequant:
        return (quantized.float() * scales[:, None, :, None]).view(rows, cols).to(x.dtype)
    return quantized.view(rows, cols), scales.to(scale_dtype)
