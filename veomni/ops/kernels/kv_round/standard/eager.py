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

"""Eager PyTorch blockwise FP8 (E4M3 + UE8M0 scale) storage round-trip."""

from __future__ import annotations

import torch
from torch import Tensor

from ....registry import SavedState


_FP8_MAX = 448.0
_FP8_MAX_INV = 1.0 / _FP8_MAX
# An all-zero block still needs a representable, non-zero scale.
_AMAX_FLOOR = 1e-4


def _ue8m0_scale(amax: Tensor) -> Tensor:
    """Round ``amax / 448`` up to a power of two, as a bare E8M0 exponent."""
    scaled = (amax * _FP8_MAX_INV).contiguous()
    bits = scaled.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    log2_ceil = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def fp8_roundtrip(x: Tensor, block_size: int = 32, quant_features: int = -1) -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` blocks with ``ue8m0`` scales.

    Blocks run along the last dimension, one scale per block per row. Every scale
    is derived from its own block's amax and rounds up to a power of two, so
    ``x / scale`` stays inside the representable range and no element is
    saturated -- which is what lets the straight-through backward be the identity.

    Args:
        x: Operand whose last dimension is the quantized dimension.
        block_size: Number of channels sharing one scale.
        quant_features: Quantize only the leading ``quant_features`` channels and
            leave the rest untouched. ``-1`` (the default) quantizes the whole
            last dimension. DeepSeek-V4 stores the NoPE half of an attention KV
            entry in FP8 and keeps the trailing RoPE channels in BF16, so the
            round trip has to stop at that split rather than run over the head.
    """
    last = x.shape[-1]
    if block_size <= 0 or last % block_size != 0:
        raise ValueError(f"last dim {last} is not divisible by FP8 block size {block_size}")
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
        quantized = fp8_roundtrip(x[..., :quant_features], block_size)
        return torch.cat([quantized, x[..., quant_features:]], dim=-1)

    orig_dtype = x.dtype
    rows = x.reshape(-1, last).float()
    n_blocks = last // block_size
    blocks = rows.reshape(rows.shape[0], n_blocks, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(_AMAX_FLOOR)
    scale = _ue8m0_scale(amax)
    quantized = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape_as(x).to(orig_dtype)


def forward(x: Tensor, *, block_size: int = 32, quant_features: int = -1) -> tuple[Tensor, SavedState]:
    """Run eager FP8 storage round-trip."""
    return fp8_roundtrip(x, block_size=block_size, quant_features=quant_features), SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
