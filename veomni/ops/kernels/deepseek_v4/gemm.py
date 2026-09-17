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
# The GEMM implementations were originally released by DeepSeek under the MIT
# License. Permission is hereby granted, free of charge, to any person
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

"""Quantized GEMMs for DeepSeek-V4 / V4.1 inference.

Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py

These are the companions to :mod:`.quant`: that module turns a bf16 activation
into fp8 plus a scale, and this one multiplies the result by a weight that was
quantized offline and never dequantized. Training does not need them -- QAT
fake-quantizes in bf16 and then calls an ordinary ``F.linear`` -- so they only
appear on the inference path, where the released checkpoint's fp8 and fp4
weights are used in the precision they ship in.

Both kernels compute ``C[M, N] = A[M, K] @ B[N, K]^T``, i.e. ``B`` is the
weight in the row-major ``[out_features, in_features]`` layout that
``nn.Linear`` already uses. Neither transposes, so no weight is ever
materialized in a second layout.
"""

import tilelang
import tilelang.language as T
import torch

from .quant import BF16, FE8M0, FP4, FP8, FP32, pass_configs


#: TileLang names for the output dtypes the kernels can write.
_OUT_DTYPE_NAMES = {torch.bfloat16: BF16, torch.float32: FP32}


def _out_dtype_name(out_dtype: torch.dtype) -> str:
    """The reference reads ``torch.get_default_dtype()`` here instead.

    That couples the result to global interpreter state: the kernel is compiled
    for bf16 while the output tensor is allocated with whatever the default
    happens to be, so a caller under ``torch.set_default_dtype(torch.float32)``
    gets a float32 buffer filled with bf16 bytes. VeOmni does not set a global
    default dtype, so the output dtype is a parameter here and the kernel is
    compiled to match it.
    """
    try:
        return _OUT_DTYPE_NAMES[out_dtype]
    except KeyError:
        raise ValueError(f"out_dtype must be one of {list(_OUT_DTYPE_NAMES)}, got {out_dtype}") from None


def _scale_dtype_name(scale_dtype: torch.dtype) -> str:
    return FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(
    N,
    K,
    block_size=128,
    out_dtype=BF16,
    accum_dtype=FP32,
    scale_dtype=FP32,
):
    assert out_dtype in [BF16, FP32]
    assert block_size in [32, 128]

    M = T.symbolic("M")
    group_size = block_size
    block_M = 32
    block_N = 128
    block_K = group_size
    num_stages = 4 if block_K == 128 else 2

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)

                # Activation scales are per row and K block. Weight scales are
                # per (N block, K block), where the block can be 32 or 128.
                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k])
                for j in T.Parallel(block_N):
                    scale_b_frag[j] = T.Cast(
                        FP32,
                        scales_b[(bx * block_N + j) // group_size, k],
                    )

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Separate accumulator for scale-corrected results.
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


@tilelang.jit(pass_configs=pass_configs)
def fp4_gemm_kernel(
    N,
    K,
    act_block_size=128,
    out_dtype=BF16,
    accum_dtype=FP32,
    scale_dtype=FP32,
):
    """FP8 act x FP4 weight GEMM kernel.

    C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T

    Act: 1x32 or 1x128 quant on K, FP8 with configurable scale dtype
    Weight: 1x32 quant on K (reduce dim), FP4 with E8M0 scale

    B is stored as [N, K//2] in float4_e2m1fn_x2, logical [N, K] in fp4.
    The FP4 values are packed along the K (last) dimension.

    Strategy: load FP4 sub-blocks of size [block_N, sub_K] (sub_K=32),
    cast FP4 to FP8 via float, then do FP8xFP8 GEMM.
    Apply activation and weight scales to the accumulator.
    """
    assert out_dtype in [BF16, FP32]
    M = T.symbolic("M")
    assert act_block_size in [32, 128]
    act_group_size = act_block_size
    weight_group_size = 32
    block_M = 32
    block_N = 128
    block_K = 32  # matches weight_group_size for simple scale handling
    n_sub = act_group_size // block_K

    @T.prim_func
    def fp4_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP4],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, act_group_size)), scale_dtype],
        scales_b: T.Tensor[(N, T.ceildiv(K, weight_group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_fp4_shared)
                # FP4->FP8 cast must go through FP32 to avoid ambiguous C++ overload
                for i, j in T.Parallel(block_N, block_K):
                    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

                # Weight scale: per 32 on K, indexed by k (each k is one block_K=32)
                for i in T.Parallel(block_N):
                    scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])

                # Activation scale: one value per act_group_size elements.
                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp4_gemm_kernel_


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``C[M, N] = A[M, K] @ B[N, K]^T`` with per-block FP8 scaling.

    ``a`` is an activation already quantized by :func:`..quant.act_quant`, so
    its scale is per row and per K block. ``b`` is a weight quantized offline,
    so its scale is per ``block_size x block_size`` tile -- one number for a
    whole square of the weight, not per row.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling factor tensors must be contiguous"
    assert block_size in (32, 128)
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % block_size == 0
    assert a_s.numel() == M * (K // block_size)
    assert b_s.shape == (
        (N + block_size - 1) // block_size,
        K // block_size,
    )
    c = a.new_empty(*a.size()[:-1], N, dtype=out_dtype)
    kernel = fp8_gemm_kernel(
        N,
        K,
        block_size=block_size,
        out_dtype=_out_dtype_name(out_dtype),
        scale_dtype=_scale_dtype_name(scale_dtype),
    )
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T``.

    A has per-32 or per-128 activation scale; B has per-32 E8M0 weight scale.
    B is stored as ``[N, K//2]`` in ``float4_e2m1fn_x2`` (two FP4 values per
    byte, packed along K).

    The mixed precision is the point: there is no fp4 activation anywhere. The
    kernel widens each fp4 weight to fp8 in shared memory and runs an ordinary
    fp8 tensor-core GEMM, so fp4 buys memory, not arithmetic.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert act_block_size in (32, 128)
    assert K % act_block_size == 0
    assert a_s.numel() == M * (K // act_block_size)
    assert b_s.shape == (N, K // 32)
    c = a.new_empty(*a.size()[:-1], N, dtype=out_dtype)
    kernel = fp4_gemm_kernel(
        N,
        K,
        act_block_size=act_block_size,
        out_dtype=_out_dtype_name(out_dtype),
        scale_dtype=_scale_dtype_name(scale_dtype),
    )
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
