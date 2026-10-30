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

"""Triton block-wise FP8 quantizers for AMD ROCm.

Drop-in counterparts of ``act_quant`` and ``fp8_weight_quant`` in
:mod:`veomni.ops.qat.quant` with the same arguments, outputs and numerics:
``amax`` floored at ``1e-4``, scale ``amax * (1 / 448)`` (optionally rounded up
to a power of two), ``x / scale`` clamped to ``±448`` and rounded to nearest
even E4M3FN. Both the stored bytes and the dequantized values are E4M3FN's on
every supported GPU, signed zeros included.
"""

import torch
import triton
import triton.language as tl

from ...utils.device import get_rocm_gcn_arch
from .scale import validate_fp8_scale_pairing


__all__ = ["act_quant", "fp8_weight_quant"]

# gfx94x only casts natively to E4M3FNUZ, whose bits encode half the E4M3FN value
# of the same bits on an identically spaced grid. Casting v / 2 there rounds
# exactly like E4M3FN, except that FNUZ has no negative zero.
_FNUZ_ARCHS = ("gfx940", "gfx941", "gfx942")

_ACT_BLOCK_M = 32


@triton.jit
def _clamped_ratio(x, amax, ROUND_SCALE: tl.constexpr):
    """Return ``(clamp(x / scale, ±448), scale)`` for the block maximum ``amax``."""
    scale = tl.maximum(amax, 1e-4) * (1.0 / 448.0)
    if ROUND_SCALE:
        bits = scale.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        # A power-of-two scale has an exact reciprocal, so this equals x / scale.
        ratio = x * ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    else:
        ratio = x / scale
    return tl.clamp(ratio, -448.0, 448.0), scale


@triton.jit
def _e4m3fn_value(v, FNUZ_CAST: tl.constexpr):
    if FNUZ_CAST:
        q = (v * 0.5).to(tl.float8e4b8).to(tl.float32) * 2.0
        sign = v.to(tl.int32, bitcast=True) & -2147483648
        return (q.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    return v.to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _e4m3fn_byte(v, FNUZ_CAST: tl.constexpr):
    if FNUZ_CAST:
        byte = (v * 0.5).to(tl.float8e4b8).to(tl.uint8, bitcast=True)
        return byte | ((v.to(tl.int32, bitcast=True) >> 24) & 0x80).to(tl.uint8)
    return v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _act_quant_kernel(
    X_PTR,
    Y_PTR,
    S_PTR,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    DEQUANT: tl.constexpr,
    E8M0_SCALE: tl.constexpr,
    FNUZ_CAST: tl.constexpr,
):
    block = tl.program_id(1)
    rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < M
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(X_PTR + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    v, scale = _clamped_ratio(x, tl.max(tl.abs(x), axis=1, keep_dims=True), ROUND_SCALE)
    if DEQUANT:
        y = (_e4m3fn_value(v, FNUZ_CAST) * scale).to(Y_PTR.dtype.element_ty)
        tl.store(Y_PTR + offsets, y, mask=row_mask[:, None])
    else:
        tl.store(Y_PTR + offsets, _e4m3fn_byte(v, FNUZ_CAST), mask=row_mask[:, None])
        row_scale = tl.reshape(scale, (BLOCK_M,))
        scale_offsets = rows * (N // BLOCK_SIZE) + block
        if E8M0_SCALE:
            biased_exponent = (row_scale.to(tl.int32, bitcast=True) >> 23) & 0xFF
            tl.store(S_PTR + scale_offsets, biased_exponent.to(tl.uint8), mask=row_mask)
        else:
            tl.store(S_PTR + scale_offsets, row_scale, mask=row_mask)


@triton.jit
def _weight_quant_kernel(
    X_PTR,
    Y_PTR,
    S_PTR,
    N,
    BLOCK_SIZE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    DEQUANT: tl.constexpr,
    E8M0_SCALE: tl.constexpr,
    FNUZ_CAST: tl.constexpr,
):
    tile_row = tl.program_id(0)
    tile_col = tl.program_id(1)
    rows = (tile_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    cols = tile_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(X_PTR + offsets).to(tl.float32)
    v, scale = _clamped_ratio(x, tl.max(tl.max(tl.abs(x), axis=1), axis=0), ROUND_SCALE)
    if DEQUANT:
        tl.store(Y_PTR + offsets, (_e4m3fn_value(v, FNUZ_CAST) * scale).to(Y_PTR.dtype.element_ty))
    else:
        tl.store(Y_PTR + offsets, _e4m3fn_byte(v, FNUZ_CAST))
        scale_offset = tile_row.to(tl.int64) * (N // BLOCK_SIZE) + tile_col
        if E8M0_SCALE:
            tl.store(S_PTR + scale_offset, ((scale.to(tl.int32, bitcast=True) >> 23) & 0xFF).to(tl.uint8))
        else:
            tl.store(S_PTR + scale_offset, scale)


def _storage(t: torch.Tensor) -> torch.Tensor:
    """Kernels write FP8 and E8M0 tensors through their raw bytes."""
    return t.view(torch.uint8) if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu) else t


def _fnuz_cast(x: torch.Tensor) -> bool:
    return get_rocm_gcn_arch(x.device) in _FNUZ_ARCHS


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    dequant: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization along the last dimension. dequant=True returns the BF16 round trip.

    When scale_fmt is set, scales are rounded up to a power of two.
    """
    N = x.size(-1)
    assert x.dtype == torch.bfloat16, f"act_quant expects a bfloat16 activation, got {x.dtype}"
    assert block_size > 0 and block_size & (block_size - 1) == 0, (
        f"block_size must be a power of two, got {block_size}"
    )
    assert N % block_size == 0, f"activation of {N} channels is not divisible by block_size {block_size}"
    validate_fp8_scale_pairing(scale_fmt, scale_dtype)
    z = x.contiguous()
    M = z.numel() // N
    y = torch.empty_like(z) if dequant else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    if M:
        _act_quant_kernel[(triton.cdiv(M, _ACT_BLOCK_M), N // block_size)](
            z,
            _storage(y),
            _storage(s),
            M,
            N,
            BLOCK_M=_ACT_BLOCK_M,
            BLOCK_SIZE=block_size,
            ROUND_SCALE=scale_fmt is not None,
            DEQUANT=dequant,
            E8M0_SCALE=scale_dtype == torch.float8_e8m0fnu,
            FNUZ_CAST=_fnuz_cast(z),
        )
    if dequant:
        return y
    return y, s


def fp8_weight_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    dequant: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Block-wise ``block_size x block_size`` FP8 quantization of a 2D weight.

    Same contract as :func:`veomni.ops.qat.quant.fp8_weight_quant`: with
    ``dequant`` the BF16 round trip alone, otherwise the E4M3FN weight and one
    ``scale_dtype`` scale per tile. ``x`` is only read.
    """
    assert x.dim() == 2, f"fp8_weight_quant expects a 2D weight, got shape {tuple(x.shape)}"
    assert x.dtype == torch.bfloat16, f"fp8_weight_quant expects a bfloat16 weight, got {x.dtype}"
    assert block_size > 0 and block_size & (block_size - 1) == 0, (
        f"block_size must be a power of two, got {block_size}"
    )
    validate_fp8_scale_pairing(scale_fmt, scale_dtype)
    M, N = x.shape
    assert M % block_size == 0 and N % block_size == 0, (
        f"weight shape {(M, N)} is not divisible by block_size {block_size}"
    )
    z = x.contiguous()
    y = torch.empty_like(z) if dequant else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty((M // block_size, N // block_size), dtype=scale_dtype)
    if M and N:
        _weight_quant_kernel[(M // block_size, N // block_size)](
            z,
            _storage(y),
            _storage(s),
            N,
            BLOCK_SIZE=block_size,
            ROUND_SCALE=scale_fmt is not None,
            DEQUANT=dequant,
            E8M0_SCALE=scale_dtype == torch.float8_e8m0fnu,
            FNUZ_CAST=_fnuz_cast(z),
            num_warps=8,
        )
    if dequant:
        return y
    return y, s
