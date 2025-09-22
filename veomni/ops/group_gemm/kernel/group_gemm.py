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

from typing import Optional

import torch
import triton
import triton.language as tl

from ....utils.device import get_torch_device
from ..utils.pretuned import algo_key_scaled, pretuned
from .triton_utils.activation import (
    ActivationType,
    activation_fwd,
)
from .triton_utils.memory import (
    load_block_with_pred_2d,
    load_with_pred_1d,
    load_with_pred_2d,
    store_block_with_pred_2d,
    store_with_pred_2d,
)
from .triton_utils.utils import (
    get_pid_mn,
    make_blocked,
)


def _get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]


# @triton.autotune(
#     configs=_get_cuda_autotune_config(),
#     key=["total_M", "N", "K"],
# )
@pretuned(
    algo_key=algo_key_scaled(["total_M", "N", "K"], [5000, 1, 1], ["TRANSPOSE_A", "TRANSPOSE_B"]),
    fallback={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP": 8},
)
@triton.heuristics(
    values={
        "N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "K_ALIGNED": lambda args: args["K"] % args["BLOCK_K"] == 0,
        "HAS_ACTIVATION": lambda args: args["ACTIVATION"] is not None,
    }
)
@triton.jit
def group_gemm_same_nk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    act_ptr,
    cumsum_M,
    max_M,
    total_M,  # Used for generating algo. key only.
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    # No need to support TRANPOSE_C, just ask user to calculate `c.t()` as `b.t() @ a.t()`.
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    K_ALIGNED: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
    SAVE_ACTIVATION: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_M + gid)
    m_size = (gtid_end - gtid_start).to(tl.uint64)

    if m * BLOCK_M >= m_size:
        return

    a_ptr += gtid_start * K
    b_ptr += gid * K * N
    c_ptr += gtid_start * N

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_am = offs_m % m_size.to(tl.int64)
    offs_bn = offs_n % N

    blk_k = tl.arange(0, BLOCK_K)

    stride_am, stride_ak = (K, 1) if not TRANSPOSE_A else (1, m_size)
    stride_bk, stride_bn = (N, 1) if not TRANSPOSE_B else (1, K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + blk_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (blk_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + N * offs_m[:, None] + 1 * offs_n[None, :]

    if ACCUMULATE_TO_C:
        c = load_with_pred_2d(
            c_ptrs,
            False,
            N_ALIGNED,
            offs_m[:, None] < m_size,
            offs_n[None, :] < N,
            other=0,
        )
    else:
        c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Really loading a 2D block. Using `load_with_pred_1d` as we only have one predicate.
        a = load_with_pred_1d(a_ptrs, K_ALIGNED, blk_k[None, :] < K - k * BLOCK_K, other=0)
        b = load_with_pred_1d(b_ptrs, K_ALIGNED, blk_k[:, None] < K - k * BLOCK_K, other=0)

        c = tl.dot(a, b, c)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_ACTIVATION:
        # Makes GELU_APPROX faster, not sure why..
        c = make_blocked(c, c_ptr.dtype.element_ty)
        if SAVE_ACTIVATION:
            store_with_pred_2d(
                act_ptr + gtid_start * N + N * offs_m[:, None] + offs_n[None, :],
                c,
                False,
                N_ALIGNED,
                offs_m[:, None] < m_size,
                offs_n[None, :] < N,
            )
        c = activation_fwd(c, ACTIVATION)

    store_with_pred_2d(c_ptrs, c, False, N_ALIGNED, offs_m[:, None] < m_size, offs_n[None, :] < N)


def group_gemm_same_nk(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    max_M: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    activation: Optional[ActivationType] = None,
    save_activation: bool = False,
    c: Optional[torch.Tensor] = None,
):
    """Grouped gemm for same nk

    Keyword arguments:
    a -- lhs matrixs to be matrix multiplied
    b -- rhs matrixs to be matrix multiplied
    cumsum_M -- matrixs's size cumsum on M
    max_M -- matrixs's max size on M
    transpose_a -- transpose `a` or not
    transpose_b -- transpose `b` or not
    activation -- activation type if needed
    save_activation -- return the activation's input or not
    c -- which tensor accumulate to, c = c + ggemm(a, b)
    """
    if transpose_b:
        G, N, K = b.shape
    else:
        G, K, N = b.shape

    assert not transpose_a, "Transpose A not tested yet."

    assert a.dtype in [torch.bfloat16, torch.float16], a.dtype
    assert b.dtype in [torch.bfloat16, torch.float16], b.dtype

    assert a.device == b.device, f"a.device = {a.device}, b.device = {b.device}"

    assert len(cumsum_M) == b.shape[0]

    assert activation is None or activation in list(ActivationType), f"Not implemented: activation is {activation}."
    assert activation or not save_activation, "Can't save activation since activation type is None"

    assert a.is_contiguous() and b.is_contiguous(), "Not implemented: Noncontiguous input."

    c_is_none = c is None
    if c_is_none:
        c = torch.empty((a.shape[1] if transpose_a else a.shape[0], N), dtype=a.dtype, device=a.device)

    if save_activation:
        act = torch.empty_like(c)

    with get_torch_device().device(a.device):
        group_gemm_same_nk_kernel[
            lambda x: (
                triton.cdiv(max_M, x["BLOCK_M"]) * triton.cdiv(N, x["BLOCK_N"]),
                x["G"],
            )
        ](
            a_ptr=a,
            b_ptr=b,
            c_ptr=c,
            act_ptr=act if save_activation else None,
            cumsum_M=cumsum_M,
            max_M=max_M,
            total_M=a.shape[0],
            G=G,
            K=K,
            N=N,
            TRANSPOSE_A=transpose_a,
            TRANSPOSE_B=transpose_b,
            ACCUMULATE_TO_C=not c_is_none,
            ACTIVATION=activation,
            SAVE_ACTIVATION=save_activation,
        )

    if save_activation:
        return c, act

    return c


# @triton.autotune(
#     configs=_get_cuda_autotune_config(),
#     key=["total_K", "M", "N"],
# )
@pretuned(
    algo_key=algo_key_scaled(["M", "N", "total_K"], [1, 1, 5000], ["TRANSPOSE_A", "TRANSPOSE_B"]),
    fallback={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP": 8},
)
@triton.heuristics(
    values={
        "M_ALIGNED": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def group_gemm_same_mn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    cumsum_K,
    total_K,  # Used for generating algo. key only.
    G: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    M_ALIGNED: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_K + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_K + gid)
    k = (gtid_end - gtid_start).to(tl.uint64)

    if TRANSPOSE_A:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + gtid_start * M,
            shape=(M, k),
            strides=(1, M),
            offsets=(m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(0, 1),
        )
    else:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + gtid_start * M,
            shape=(M, k),
            strides=(k, 1),
            offsets=(m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
    if TRANSPOSE_B:
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + gtid_start * N,
            shape=(k, N),
            strides=(1, k),
            offsets=(0, n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
    else:
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + gtid_start * N,
            shape=(k, N),
            strides=(N, 1),
            offsets=(0, n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + gid * M * N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(m * BLOCK_M, n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Special case: no GEMM needed.
    if k == 0:
        if not ACCUMULATE_TO_C:
            # Zero out the corresponding output region.
            store_block_with_pred_2d(
                c_block_ptr,
                # tl.zeros(..., dtype=c_block_ptr.dtype.element_ty) raises "not implemented".
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(c_block_ptr.dtype.element_ty),
                M_ALIGNED,
                N_ALIGNED,
            )
        else:
            # Nothing to do then, just leave the kernel.
            pass

        return

    if ACCUMULATE_TO_C:
        out = tl.load(c_block_ptr).to(tl.float32)
    else:
        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # FIXME: Weird type conversion.
    for _ in range(tl.cdiv(k.to(tl.int64), BLOCK_K)):
        a = load_block_with_pred_2d(a_block_ptr, M_ALIGNED, False)
        b = load_block_with_pred_2d(b_block_ptr, False, N_ALIGNED)

        out += tl.dot(a, b)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    store_block_with_pred_2d(c_block_ptr, out.to(c_block_ptr.dtype.element_ty), M_ALIGNED, N_ALIGNED)


def group_gemm_same_mn(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    cumsum_K: torch.Tensor,
    max_K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
    G, M, N = c.shape

    assert a.dtype in [torch.bfloat16, torch.float16], a.dtype
    assert b.dtype in [torch.bfloat16, torch.float16], b.dtype

    assert a.device == b.device, f"a.device = {a.device}, b.device = {b.device}"
    assert a.device == c.device, f"a.device = {a.device}, c.device = {c.device}"

    # TODO(wenyawei):
    assert c is not None, c
    assert len(cumsum_K) == c.shape[0], f"{len(cumsum_K), c.shape}"
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous(), "Not implemented: Noncontiguous input."

    with get_torch_device().device(a.device):
        group_gemm_same_mn_kernel[
            lambda x: (
                triton.cdiv(M, x["BLOCK_M"]) * triton.cdiv(N, x["BLOCK_N"]),
                x["G"],
            )
        ](
            a_ptr=a,
            b_ptr=b,
            c_ptr=c,
            cumsum_K=cumsum_K,
            total_K=b.shape[0],
            G=G,
            M=M,
            N=N,
            TRANSPOSE_A=transpose_a,
            TRANSPOSE_B=transpose_b,
            ACCUMULATE_TO_C=False,
        )
