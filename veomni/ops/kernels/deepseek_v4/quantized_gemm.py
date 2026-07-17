# Copyright (c) 2023 DeepSeek
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
#
# Adapted from DeepSeek-V4's official inference kernel.

"""Quantized GEMMs used by the DeepSeek-V4 reference inference path."""

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


@tilelang.jit(pass_configs=pass_configs)
def fp4_gemm_kernel(n: int, k: int, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    """FP8 activation by packed FP4 weight GEMM with reference block scaling."""
    m = T.symbolic("m")
    act_group_size = 128
    weight_group_size = 32
    block_m = 32
    block_n = 128
    block_k = 32
    act_subblocks = act_group_size // block_k

    @T.prim_func
    def kernel(
        a: T.Tensor[(m, k), FP8],
        b: T.Tensor[(n, k), FP4],
        c: T.Tensor[(m, n), out_dtype],
        scales_a: T.Tensor[(m, T.ceildiv(k, act_group_size)), scale_dtype],
        scales_b: T.Tensor[(n, T.ceildiv(k, weight_group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), FP8)
            b_fp4_shared = T.alloc_shared((block_n, block_k), FP4)
            b_shared = T.alloc_shared((block_n, block_k), FP8)
            c_shared = T.alloc_shared((block_m, block_n), out_dtype)
            c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            c_accum = T.alloc_fragment((block_m, block_n), accum_dtype)
            scale_a = T.alloc_fragment((block_m,), FP32)
            scale_b = T.alloc_fragment((block_n,), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(c_local)
            T.clear(c_accum)

            for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=2):
                T.copy(a[by * block_m, ki * block_k], a_shared)
                T.copy(b[bx * block_n, ki * block_k], b_fp4_shared)
                for i, j in T.Parallel(block_n, block_k):
                    b_shared[i, j] = T.Cast(FP8, T.Cast(FP32, b_fp4_shared[i, j]))
                for i in T.Parallel(block_n):
                    scale_b[i] = T.Cast(FP32, scales_b[bx * block_n + i, ki])
                for i in T.Parallel(block_m):
                    scale_a[i] = T.Cast(FP32, scales_a[by * block_m + i, ki // act_subblocks])

                T.gemm(a_shared, b_shared, c_local, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    c_accum[i, j] += c_local[i, j] * scale_a[i] * scale_b[j]
                T.clear(c_local)

            T.copy(c_accum, c_shared)
            T.copy(c_shared, c[by * block_m, bx * block_n])

    return kernel


def fp4_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute ``A[M,K] @ B[N,K].T`` from FP8 activations and packed FP4 weights."""
    if not all(tensor.is_contiguous() for tensor in (a, a_scale, b, b_scale)):
        raise ValueError("DeepSeek-V4 quantized GEMM inputs and scales must be contiguous")
    if a.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Expected FP8 E4M3 activations, got {a.dtype}")
    if b.dtype != torch.float4_e2m1fn_x2:
        raise TypeError(f"Expected packed FP4 E2M1 weights, got {b.dtype}")

    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    if b.size(-1) * 2 != k:
        raise ValueError(f"Incompatible activation/weight K dimensions: {k} and {b.size(-1) * 2}")
    output = a.new_empty(*a.shape[:-1], n, dtype=torch.bfloat16)
    tl_scale_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    kernel = fp4_gemm_kernel(n, k, scale_dtype=tl_scale_dtype)
    kernel(a.view(m, k), b, output.view(m, n), a_scale.view(m, -1), b_scale)
    return output
