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

"""Triton JIT kernels for the FP8 and FP4 KV/indexer storage round-trips."""

import triton
import triton.language as tl


@triton.jit
def _triton_fp8_roundtrip_kernel(
    X,
    Y,
    X_STRIDE_ROW,
    Y_STRIDE_ROW,
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    col_offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * X_STRIDE_ROW + col_offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
    scaled = amax * (1.0 / 448.0)
    bits = scaled.to(tl.int32, bitcast=True)
    exp_byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exp_byte << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
    tl.store(Y + row * Y_STRIDE_ROW + col_offs, (q * scale).to(Y.dtype.element_ty))


@triton.jit
def _triton_fp4_roundtrip_kernel(
    X,
    Y,
    X_STRIDE_ROW,
    Y_STRIDE_ROW,
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    USE_E4M3_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1).to(tl.int64)
    col_offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * X_STRIDE_ROW + col_offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    if USE_E4M3_SCALE:
        # ``div_rn`` rather than ``/``: Triton's default fp32 division is not
        # IEEE round-to-nearest, and ``e4m3`` scales are not powers of two (three
        # mantissa bits), so a 1-ulp difference here lands on one of the E2M1
        # midpoints below and flips the rounded magnitude. ``ue8m0`` scales are
        # powers of two, where any division is exact -- which is why only the
        # ``e4m3`` case diverged. The eager row divides with ``div.rn.f32``.
        scale = tl.div_rn(tl.maximum(amax, 6.0 * (2.0**-9)), 6.0)
        scale = tl.minimum(scale, 448.0)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
    else:
        # Smallest E8M0 exponent step, then round the ratio up to a power of two.
        scaled = tl.maximum(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
        bits = scaled.to(tl.int32, bitcast=True)
        exp_byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exp_byte << 23).to(tl.float32, bitcast=True)
    # Same reason as the scale above: ``div_rn`` keeps ``x / scale`` identical to
    # the eager row's ``div.rn.f32`` for a non-power-of-two scale.
    v = tl.div_rn(x, scale)
    v = tl.minimum(tl.maximum(v, -6.0), 6.0)
    a = tl.abs(v)
    # Nearest E2M1 magnitude; a value exactly on a midpoint rounds down.
    nearest = tl.where(
        a <= 1.75,
        tl.where(
            a <= 0.75,
            tl.where(a <= 0.25, 0.0, 0.5),
            tl.where(a <= 1.25, 1.0, 1.5),
        ),
        tl.where(
            a <= 3.5,
            tl.where(a <= 2.5, 2.0, 3.0),
            tl.where(a <= 5.0, 4.0, 6.0),
        ),
    )
    # Preserve the input's sign bit, including ``-0.0``. ``v < 0`` alone makes a
    # small negative value store ``+0``, and the release keeps ``-0`` as its own
    # E2M1 codepoint. Reading the bit directly is the same test the eager row's
    # ``torch.signbit`` makes, and it needs no libdevice symbol.
    negative = v.to(tl.int32, bitcast=True) < 0
    rounded = tl.where(negative, -nearest, nearest)
    tl.store(Y + row * Y_STRIDE_ROW + col_offs, (rounded * scale).to(Y.dtype.element_ty))
