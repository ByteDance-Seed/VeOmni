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


FP4_MAX = 6.0
#: Smallest block amax a scale is derived from, per scale format. An all-zero
#: block still needs a finite scale, and the floor is the format's own smallest
#: normal (E4M3) or smallest exponent (E8M0) step rather than an arbitrary one.
_E4M3_AMAX_FLOOR = FP4_MAX * (2**-9)
_UE8M0_AMAX_FLOOR = FP4_MAX * (2**-126)
#: Largest finite E4M3 value. ``Tensor.to(torch.float8_e4m3fn)`` is a *narrowing*
#: cast, not a saturating one: a value above this becomes NaN, and a NaN scale
#: poisons its whole block. The ratio therefore has to be clamped before the cast,
#: which is what makes a block whose amax exceeds ``6 * 448`` clamp at all instead
#: of vanishing.
_E4M3_MAX_RATIO = float(torch.finfo(torch.float8_e4m3fn).max)
SCALE_FORMATS = ("e4m3", "ue8m0")


def _round_to_fp4(values: Tensor) -> Tensor:
    """Round to the nearest E2M1 magnitude, keeping the sign *including its zero*.

    The midpoints between consecutive magnitudes are ``0.25, 0.75, 1.25, 1.75,
    2.5, 3.5, 5.0``; a value exactly on one rounds *down*. That is what the
    released DeepSeek checkpoints were produced with and what the deployment
    kernel reads back, so it is the convention here -- note it is not
    round-half-to-even (``0.75 -> 0.5``, ``2.5 -> 2.0``), which is what a
    hardware E2M1 convert typically does. Written out rather than derived from a
    table because the table would have to be indexed by an already-quantized code.

    The sign comes from the magnitude of the *encoded* value via ``signbit``
    rather than ``sign()``: ``sign()`` returns ``0`` for ``-0.0``, so a small
    negative input would be stored as ``+0`` and lose the distinction the release
    preserves. ``-0.0`` is a real E2M1 codepoint (bit pattern ``0x8``), so it has
    to survive the round trip.
    """
    a = values.abs()
    nearest = torch.where(
        a <= 1.75,
        torch.where(
            a <= 0.75,
            torch.where(a <= 0.25, 0.0, 0.5),
            torch.where(a <= 1.25, 1.0, 1.5),
        ),
        torch.where(
            a <= 3.5,
            torch.where(a <= 2.5, 2.0, 3.0),
            torch.where(a <= 5.0, 4.0, 6.0),
        ),
    )
    return torch.where(torch.signbit(values), -nearest, nearest)


def _pow2_scale(amax: Tensor) -> Tensor:
    """Round ``amax / 6`` up to a power of two, as a bare E8M0 exponent."""
    ratio = (amax * (1.0 / FP4_MAX)).clamp_min(torch.finfo(torch.float32).tiny).contiguous()
    bits = ratio.view(torch.int32)
    exp_byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exp_byte << 23).view(torch.float32)


def fp4_roundtrip(x: Tensor, block_size: int = 16, scale_fmt: str = "e4m3") -> Tensor:
    """Quantize-dequantize ``x`` in ``1 x block_size`` groups to E2M1 FP4.

    One scale per block per row along the last dimension. Unlike the FP8 round
    trip this saturates: E2M1 tops out at 6, so a block whose amax rounds down
    (an ``e4m3`` scale is not a power of two and rounds to nearest, so ``6 *
    scale`` can land below the amax) clamps its largest entries. The
    straight-through backward is still what a deployment kernel effectively does
    -- it reads back the rounded values and multiplies them.

    The round trip is idempotent for ``ue8m0`` scales, whose power-of-two step
    keeps ``scale(round(x)) == scale(x)``, but **not** for every ``e4m3`` block:
    near the subnormal floor an already-rounded value can select the next scale
    down (``39/2048 -> 15.6/1024 -> 12/1024``). That is a property of rounding a
    scale to three mantissa bits, not a bug, and it matches what re-encoding a
    stored tensor through the reference quantizer does. Callers treat this as an
    encode, not a lossless transform -- which is how training uses it.
    """
    if scale_fmt not in SCALE_FORMATS:
        raise ValueError(f"scale_fmt must be one of {SCALE_FORMATS}, got {scale_fmt!r}")
    last = x.shape[-1]
    if block_size <= 0 or last % block_size != 0:
        raise ValueError(f"last dim {last} is not divisible by FP4 group size {block_size}")
    if x.numel() == 0:
        return x
    orig_dtype = x.dtype
    rows = x.reshape(-1, last).float()
    n_blocks = last // block_size
    blocks = rows.reshape(rows.shape[0], n_blocks, block_size)
    amax = blocks.abs().amax(-1, keepdim=True)
    if scale_fmt == "e4m3":
        # Keep this arithmetic identical to the Triton kernel's: floor the amax,
        # *divide* by 6 (a reciprocal multiply differs in fp32 for about a third of
        # inputs and the two rows are compared bitwise), then clamp before the
        # narrowing cast.
        ratio = (amax.clamp_min(_E4M3_AMAX_FLOOR) / FP4_MAX).clamp(max=_E4M3_MAX_RATIO)
        scale = ratio.to(torch.float8_e4m3fn).float()
    else:
        scale = _pow2_scale(amax.clamp_min(_UE8M0_AMAX_FLOOR))
    quantized = _round_to_fp4((blocks / scale).clamp(-FP4_MAX, FP4_MAX))
    return (quantized * scale).reshape_as(x).to(orig_dtype)


def forward(x: Tensor, *, block_size: int = 16, scale_fmt: str = "e4m3") -> tuple[Tensor, SavedState]:
    """Run eager FP4 storage round-trip."""
    return fp4_roundtrip(x, block_size=block_size, scale_fmt=scale_fmt), SavedState(())


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor]:
    """Pass gradient straight through to the input."""
    del saved
    return (grad_output,)
