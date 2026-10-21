# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Deterministic batched matmul Triton kernel used by DeepSeek V3's deterministic RoPE path.

Originally adapted from https://github.com/thinking-machines-lab/batch_invariant_ops.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N)"""
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)
    if pid_b >= B:
        return
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)
    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_batch_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_batch_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_valid[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_valid[:, None] & mask_n[None, :], other=0.0)
        accumulator = tl.dot(a, b, accumulator)
    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)
    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Deterministic batched matmul via Triton: (B, M, K) x (B, K, N) -> (B, M, N)"""
    assert a.ndim == 3 and b.ndim == 3, f"Expected 3D tensors, got {a.ndim}D and {b.ndim}D"
    B, M, K = a.shape
    _, K2, N = b.shape
    assert K == K2 and a.shape[0] == b.shape[0]
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128))
    BLOCK_N = 16 if N <= 16 else (32 if N <= 32 else (64 if N <= 64 else 128))
    BLOCK_K = 16 if K <= 16 else (32 if K <= 32 else 64)
    grid = (B, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    _bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
    )
    return c


__all__ = ["triton_bmm"]
