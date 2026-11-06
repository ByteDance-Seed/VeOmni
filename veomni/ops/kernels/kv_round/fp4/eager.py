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

"""Eager PyTorch blockwise FP4 (E2M1 + E4M3/UE8M0 scale) storage round-trip."""

from __future__ import annotations

import torch
from torch import Tensor

from ....registry import SavedState


_FP4_MAX = 6.0
# E2M1 codepoints. ``to_fp4`` picks the nearest of these, which is what a
# hardware cast does; there is no stochastic rounding to reproduce.
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# An all-zero group still needs a representable, non-zero scale.
_FP4_E4M3_AMAX_FLOOR = _FP4_MAX * (2**-9)
_FP4_E8M0_AMAX_FLOOR = _FP4_MAX * (2**-126)


def _round_to_fp4(values: Tensor) -> Tensor:
    """Round magnitudes to the nearest E2M1 codepoint and restore the sign."""
    levels = values.new_tensor(_FP4_LEVELS)
    magnitude = values.abs()
    nearest = levels[(magnitude.unsqueeze(-1) - levels).abs().argmin(-1)]
    return values.sign() * nearest


def _pow2_scale(amax: Tensor) -> Tensor:
    """Round ``amax / 6`` up to a power of two (E8M0), by integer exponent arithmetic."""
    ratio = (amax * (1.0 / _FP4_MAX)).clamp_min(torch.finfo(torch.float32).tiny).contiguous()
    bits = ratio.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent << 23).view(torch.float32)


def fp4_roundtrip(x: Tensor, block_size: int = 16, scale_fmt: str = "e4m3", quant_features: int = -1) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to E2M1.

    Args:
        x: Operand whose last dimension is the quantized dimension.
        block_size: Number of channels sharing one scale.
        scale_fmt: ``"e4m3"`` stores a signed FP8 scale; ``"ue8m0"`` rounds it up
            to a power of two so it fits in a bare E8M0 exponent.
        quant_features: Quantize only the leading ``quant_features`` channels.
            ``-1`` (the default) quantizes the whole last dimension.
    """
    if scale_fmt not in ("e4m3", "ue8m0"):
        raise ValueError(f"scale_fmt must be 'e4m3' or 'ue8m0', got {scale_fmt!r}")
    last = x.shape[-1]
    if block_size <= 0 or last % block_size != 0:
        raise ValueError(f"last dim {last} is not divisible by FP4 group size {block_size}")
    if x.numel() == 0:
        return x
    if quant_features != -1:
        if not 0 < quant_features <= last:
            raise ValueError(f"quant_features must fall in (0, {last}], got {quant_features}")
        if quant_features % block_size:
            raise ValueError(f"quant_features {quant_features} is not divisible by block_size {block_size}")
        if quant_features == last:
            quant_features = -1
    if quant_features != -1:
        quantized = fp4_roundtrip(x[..., :quant_features], block_size, scale_fmt)
        return torch.cat([quantized, x[..., quant_features:]], dim=-1)

    orig_dtype = x.dtype
    rows = x.reshape(-1, last).float()
    n_blocks = last // block_size
    blocks = rows.reshape(rows.shape[0], n_blocks, block_size)
    amax = blocks.abs().amax(-1, keepdim=True)
    if scale_fmt == "e4m3":
        amax = amax.clamp_min(_FP4_E4M3_AMAX_FLOOR)
        scale = (amax / _FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(_FP4_E8M0_AMAX_FLOOR)
        scale = _pow2_scale(amax)
    quantized = _round_to_fp4((blocks / scale).clamp(-_FP4_MAX, _FP4_MAX))
    return (quantized * scale).reshape_as(x).to(orig_dtype)


def forward(
    x: Tensor, *, block_size: int = 16, scale_fmt: str = "e4m3", quant_features: int = -1
) -> tuple[Tensor, SavedState]:
    """Run eager FP4 storage round-trip."""
    return fp4_roundtrip(x, block_size=block_size, scale_fmt=scale_fmt, quant_features=quant_features), SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
