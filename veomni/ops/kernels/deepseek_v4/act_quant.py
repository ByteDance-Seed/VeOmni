# Copyright (c) 2023 DeepSeek
# Copyright 2026 the Miles contributors and ByteDance Ltd. and/or its affiliates
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
#
# The act_quant implementation was originally released by DeepSeek under the
# MIT License. Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, subject to inclusion of this
# copyright and permission notice. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
# WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Adapted through radixark/miles; modified for VeOmni.

"""Block-wise FP8 activation quantization for DeepSeek-V4.

Ported verbatim from deepseek-ai/DeepSeek-V4-Pro/inference/kernel.py to keep
bit-exact parity with the upstream inference kernel. Keep this file in sync
when DeepSeek updates the reference implementation.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py
"""

import tilelang
import tilelang.language as T
import torch


tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def fast_log2_ceil(x):
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False, inplace=False
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    # Internal computation in FP32; scale_dtype controls output storage format.
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP8, T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)))
                            * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    if pid_m * blk_m + i < M:
                        S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP).
    """
    N = x.size(-1)
    assert N % block_size == 0
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    kernel = act_quant_kernel(
        N,
        block_size,
        scale_dtype=tl_dtype,
        round_scale=scale_fmt is not None,
        inplace=inplace,
    )
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp8_weight_quant_kernel(
    M,
    N,
    block_size=128,
    scale_dtype=FP32,
    round_scale=False,
):
    """Block-wise ``block_size x block_size`` FP8 quantization, one tile per CTA.
    round_scale=True rounds each tile scale up to a power of two (MXFP).
    """
    assert M % block_size == 0 and N % block_size == 0
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max

    @T.prim_func
    def fp8_weight_quant_kernel_(
        X: T.Tensor[(M, N), BF16],
        Y: T.Tensor[(M, N), FP8],
        S: T.Tensor[(M // block_size, N // block_size), scale_dtype],
    ):
        with T.Kernel(N // block_size, M // block_size, threads=128) as (bx, by):
            x_shared = T.alloc_shared((block_size, block_size), BF16)
            y_shared = T.alloc_shared((block_size, block_size), FP8)
            amax_row_local = T.alloc_fragment((block_size,), FP32)
            scale_local = T.alloc_fragment((1,), FP32)
            scale_shared = T.alloc_shared((1,), FP32)

            T.copy(X[by * block_size, bx * block_size], x_shared)
            T.reduce_absmax(x_shared, amax_row_local, dim=1)
            T.reduce_absmax(amax_row_local, scale_local, dim=0)
            scale_local[0] = T.max(scale_local[0], 1e-4)
            if round_scale:
                scale_local[0] = fast_round_scale(scale_local[0], fp8_max_inv)
            else:
                scale_local[0] = scale_local[0] * fp8_max_inv
            T.copy(scale_local, scale_shared)

            for i, j in T.Parallel(block_size, block_size):
                y_shared[i, j] = T.clamp(x_shared[i, j] / scale_shared[0], fp8_min, fp8_max)

            T.copy(y_shared, Y[by * block_size, bx * block_size])
            if T.get_thread_binding(0) == 0:
                S[by, bx] = T.Cast(scale_dtype, scale_shared[0])

    return fp8_weight_quant_kernel_


def fp8_weight_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise ``block_size x block_size`` FP8 quantization of a 2D weight.

    Args:
        x (torch.Tensor): The bfloat16 weight to quantize.
        block_size (int): Side length of the square quantization tile.
        scale_fmt (str | None): When set, tile scales are rounded up to a power
            of two, as the MXFP formats require. DeepSeek V4 uses ``"ue8m0"``.
        scale_dtype (torch.dtype): Storage dtype of the returned scales, either
            ``torch.float32`` or ``torch.float8_e8m0fnu``. E8M0 carries no
            mantissa, so it is only usable together with ``scale_fmt``.

    Returns:
        weight (torch.float8_e4m3fn): Quantized weight, same shape as ``x``.
        scale (``scale_dtype``): One scale per ``block_size x block_size`` tile.
    """
    assert x.dim() == 2, f"fp8_weight_quant expects a 2D weight, got shape {tuple(x.shape)}"
    assert x.dtype == torch.bfloat16, f"fp8_weight_quant expects a bfloat16 weight, got {x.dtype}"
    assert scale_dtype in (torch.float32, torch.float8_e8m0fnu), (
        f"fp8_weight_quant supports float32 and float8_e8m0fnu scales, got {scale_dtype}"
    )
    # Without rounding, the stored E8M0 scale would differ from the FP32 scale
    # the kernel divides by, so dequantization would silently drift.
    assert scale_dtype != torch.float8_e8m0fnu or scale_fmt is not None, (
        'float8_e8m0fnu scales only represent powers of two: pass scale_fmt (DeepSeek V4 uses "ue8m0")'
    )
    M, N = x.shape
    assert M % block_size == 0 and N % block_size == 0, (
        f"weight shape {(M, N)} is not divisible by block_size {block_size}"
    )
    z = x.contiguous()
    y = torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty((M // block_size, N // block_size), dtype=scale_dtype)
    kernel = fp8_weight_quant_kernel(
        M,
        N,
        block_size,
        scale_dtype=scale_dtype,
        round_scale=scale_fmt is not None,
    )
    kernel(z, y, s)
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp4_quant_kernel(N, block_size=32, in_dtype=BF16, scale_dtype=FE8M0, inplace=False):
    """Block-wise FP4 quantization. Power-of-2 scale via bit ops. inplace=True does fused quant+dequant."""
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32
    group_size = block_size
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else FP4

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP4, T.clamp(x_local[i, j] / s_local[i], -fp4_max, fp4_max)))
                            * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], -fp4_max, fp4_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_quant_kernel_


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP4 quantization. inplace=True does fused quant+dequant back to BF16."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
    kernel = fp4_quant_kernel(N, block_size, inplace=inplace)
    kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s
