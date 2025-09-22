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


import torch
import triton
import triton.language as tl

from ....utils.device import get_torch_device
from .triton_utils.memory import (
    load_with_pred_1d,
    store_with_pred_1d,
)


@triton.heuristics(values={"BLOCK_ALIGNED": lambda args: args["num_elts"] % args["BLOCK_SIZE"] == 0})
@triton.jit
def _expert_histogram_kernel(
    out_ptr,
    x_ptr,
    num_elts,
    num_bins,
    NUM_BINS_LAST_UNUSED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)

    in_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = load_with_pred_1d(x_ptr + in_off, BLOCK_ALIGNED, in_off < num_elts, NUM_BINS_LAST_UNUSED - 1).to(tl.int32)

    tl.device_assert(
        data < num_bins or data == NUM_BINS_LAST_UNUSED - 1,
        "Out-of-bound element found.",
    )
    count = tl.histogram(data, NUM_BINS_LAST_UNUSED)

    out_off = tl.arange(0, NUM_BINS_LAST_UNUSED)
    tl.atomic_add(out_ptr + out_off, count, mask=out_off < num_bins, sem="relaxed")


def expert_histogram(input: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Returns histogram of `input`, with bin width 1. Note that for each individual `num_bins`,
    a separate Triton kernel is generated (mostly). So if `num_bins` varies between calls, you
    probably should go for some other histogram method.
    """

    assert input.is_cuda
    assert input.dtype == torch.int32 or input.dtype == torch.int64
    assert input.numel() < (1 << 31) - 1, "Too many elements."
    flattened = input.flatten().contiguous()

    # An extra slot is needed, our kernel uses the extra slot to handle possible OoO reads.
    # Wastes a lot of slots but hopefully the kernel can still saturate memory B/W.
    NUM_BINS_LAST_UNUSED = triton.next_power_of_2(num_bins + 1)
    out = torch.zeros([num_bins], dtype=torch.int32, device=input.device)

    BLOCK_SIZE = 1024
    num_elts = flattened.numel()
    grid = (triton.cdiv(num_elts, BLOCK_SIZE),)
    with get_torch_device().device(input.device):
        _expert_histogram_kernel[grid](
            out_ptr=out,
            x_ptr=flattened,
            num_elts=num_elts,
            num_bins=num_bins,
            NUM_BINS_LAST_UNUSED=NUM_BINS_LAST_UNUSED,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out[:num_bins]


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _moe_gather_kernel(
    X,
    Y,
    index,
    num_elts_in,
    num_elts_out,
    N: tl.constexpr,  # hidden size
    TOPK: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XN: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    r"""
    X: m * topk x n
    Y: m x n
    index: m x topk
    code:
        repeated-X: m * topk x n -> reduce(sum_over_topk) -> m x n
        Y: Y[arange(m)] = sum_over_topk(repeated-X[arange(m) * topk])
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)  # m
    block_idx = tl.program_id(axis=1).to(tl.int64)
    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    y = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        x_index = tl.load(index + pid_m.to(tl.int64) * STRIDE_IM + i * STRIDE_IN)
        tl.device_assert(x_index < num_elts_in, "Input OOB")
        x = load_with_pred_1d(
            X + x_index.to(tl.int64) * STRIDE_XM + n.to(tl.int64) * STRIDE_XN, N_ALIGNED, mask=n < N, other=0
        )
        y += x
    # save one line
    tl.device_assert(pid_m < num_elts_out, "Output OOB")
    Y = Y + pid_m.to(tl.int64) * STRIDE_OM + n.to(tl.int64) * STRIDE_ON  # noqa
    store_with_pred_1d(Y, y, N_ALIGNED, mask=n < N)


def moe_gather(x: torch.Tensor, index: torch.Tensor, out_dtype=None):
    assert x.is_cuda and index.is_cuda
    M, topk = index.shape
    assert x.shape[0] == M * topk
    N = x.shape[1]

    assert x.device == index.device, f"x.device = {x.device}, index.device = {index.device}"

    out_dtype = out_dtype or x.dtype
    out = torch.empty(M, N, dtype=out_dtype, device=x.device)

    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa
    with get_torch_device().device(x.device):
        _moe_gather_kernel[grid](
            x,
            out,
            index,
            num_elts_in=M * topk,
            num_elts_out=M,
            N=N,
            TOPK=topk,
            STRIDE_XM=x.stride(0),
            STRIDE_XN=x.stride(1),
            STRIDE_OM=out.stride(0),
            STRIDE_ON=out.stride(1),
            STRIDE_IM=index.stride(0),
            STRIDE_IN=index.stride(1),
            BLOCK_N=1024,
        )

    return out


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _moe_add_gather_kernel(
    X,
    Y,
    Z,
    index,
    num_elts_in,
    num_elts_out,
    N: tl.constexpr,  # hidden size
    TOPK: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XN: tl.constexpr,
    STRIDE_YM: tl.constexpr,
    STRIDE_YN: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    r"""
    X: m * topk x n
    Y: m * topk x n
    Z: m x n
    index: m x topk

    code:
        repeated-(X + Y): m * topk x n -> reduce(sum_over_topk) -> m x n
        Z: Z[arange(m)] = sum_over_topk(repeated-(X+Y)[arange(m) * topk])
    """
    pid_m = tl.program_id(axis=0)  # m
    block_idx = tl.program_id(axis=1)

    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    z = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        x_index = tl.load(index + pid_m * STRIDE_IM + i * STRIDE_IN)
        tl.device_assert(x_index < num_elts_in, "Input OOB")
        x = load_with_pred_1d(X + x_index * STRIDE_XM + n * STRIDE_XN, N_ALIGNED, mask=n < N, other=0)
        y = load_with_pred_1d(Y + x_index * STRIDE_YM + n * STRIDE_YN, N_ALIGNED, mask=n < N, other=0)
        z += x + y

    # save one line
    tl.device_assert(pid_m < num_elts_out, "Output OOB")
    Z = Z + pid_m * STRIDE_OM + n * STRIDE_ON  # noqa
    store_with_pred_1d(Z, z, N_ALIGNED, mask=n < N)


def moe_add_gather(x: torch.Tensor, y: torch.Tensor, index: torch.Tensor, out_dtype=None):
    assert x.is_cuda and y.is_cuda and index.is_cuda
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    M, topk = index.shape
    assert x.shape[0] == M * topk
    N = x.shape[1]

    assert x.device == y.device, f"x.device = {x.device}, y.device = {y.device}"
    assert x.device == index.device, f"x.device = {x.device}, index.device = {index.device}"

    out_dtype = out_dtype or x.dtype
    out = torch.empty(M, N, dtype=out_dtype, device=x.device)

    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa
    with get_torch_device().device(x.device):
        _moe_add_gather_kernel[grid](
            x,
            y,
            out,
            index,
            num_elts_in=M * topk,
            num_elts_out=M,
            N=N,
            TOPK=topk,
            STRIDE_XM=x.stride(0),
            STRIDE_XN=x.stride(1),
            STRIDE_YM=y.stride(0),
            STRIDE_YN=y.stride(1),
            STRIDE_OM=out.stride(0),
            STRIDE_ON=out.stride(1),
            STRIDE_IM=index.stride(0),
            STRIDE_IN=index.stride(1),
            BLOCK_N=1024,
        )

    return out


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _moe_scatter_kernel(
    X,
    O,  # noqa
    index,
    num_elts_in,
    num_elts_out,
    N: tl.constexpr,  # hidden size
    TOPK: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XN: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    r"""
    X: m x n
    O: m * topk x n
    index: m x topk

    code:
        X: m x n -> repeat -> m x topk x n -> m * topk x n
            X[arange(m) * topk] = X[arange(m)]

        O[index] = X
            O[index[arange(m) * topk]] = X[arange(m) * topk]
    """

    pid_m = tl.program_id(axis=0)  # m
    block_idx = tl.program_id(axis=1)
    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.device_assert(pid_m < num_elts_in, "Input OOB.")
    X = X + pid_m * STRIDE_XM + n * STRIDE_XN
    x = load_with_pred_1d(X, N_ALIGNED, mask=n < N, other=0)

    for i in tl.static_range(TOPK):
        o_index = tl.load(index + pid_m * STRIDE_IM + i * STRIDE_IN)
        tl.device_assert(o_index < num_elts_out, "Output OOB.")
        tmp_index = o_index.to(tl.int64) * STRIDE_OM
        # tl.device_print("tmp_index", tmp_index)
        out = O + tmp_index + n * STRIDE_ON

        # save one line
        store_with_pred_1d(out, x, N_ALIGNED, mask=n < N)


def moe_scatter(x: torch.Tensor, index: torch.Tensor, out_dtype=None):
    assert x.is_cuda and index.is_cuda
    assert x.shape[0] == index.shape[0]

    assert x.device == index.device, f"x.device = {x.device}, index.device = {index.device}"

    M, N = x.shape
    topk = index.shape[1]
    out_dtype = out_dtype or x.dtype
    out = torch.empty(M * topk, N, dtype=out_dtype, device=x.device)
    assert lambda: index.unique().numel() == M * topk, "Holes in output?"

    grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa
    with get_torch_device().device(x.device):
        _moe_scatter_kernel[grid](
            x,
            out,
            index,
            num_elts_in=M,
            num_elts_out=M * topk,
            N=N,
            TOPK=topk,
            STRIDE_XM=x.stride(0),
            STRIDE_XN=x.stride(1),
            STRIDE_OM=out.stride(0),
            STRIDE_ON=out.stride(1),
            STRIDE_IM=index.stride(0),
            STRIDE_IN=index.stride(1),
            BLOCK_N=1024,
        )

    return out


@triton.jit
def _moe_index_compute_kernel(
    indices_ptr,
    experts_for_tokens_ptr,
    temp_histogram_cumsum_ptr,
    num_elts,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Unlikely to be aligned, so we don't test for alignment.
):
    _OOB_EXPERT_ID: tl.constexpr = 1023
    tl.static_assert(_OOB_EXPERT_ID > NUM_EXPERTS, "Too many experts for me.")

    start_pos = tl.program_id(0)
    processing_range = start_pos * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_ids = tl.load(
        experts_for_tokens_ptr + processing_range,
        processing_range < num_elts,
        _OOB_EXPERT_ID,
    )
    assert expert_ids < NUM_EXPERTS or expert_ids == _OOB_EXPERT_ID

    indices = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for expert_id in tl.static_range(NUM_EXPERTS):
        mask = expert_ids == expert_id
        one_if_expert_id_matches = mask.to(tl.int32)

        # Tokens allocated to this expert.
        slots_to_reserve = tl.sum(one_if_expert_id_matches)
        slot_ids = (
            # Reserve last `slots_to_reserve` slots for us.
            tl.atomic_add(temp_histogram_cumsum_ptr + expert_id, -slots_to_reserve, sem="relaxed")
            # `atomic_add` returns old value, so we need to do substraction again.
            - slots_to_reserve
            # Local offset for each token in `expert_ids`.
            + tl.cumsum(one_if_expert_id_matches)
            # Result of `cumsum` is "1-based".
            - 1
        )
        assigned_slot_or_zero = tl.where(mask, slot_ids, 0)
        indices += assigned_slot_or_zero.to(tl.int32)

    tl.store(indices_ptr + processing_range, indices, processing_range < num_elts)


def moe_index_compute(experts_for_tokens: torch.Tensor, expert_histogram_cumsum: torch.Tensor) -> torch.Tensor:
    """Calculate row number into activation passed to MoE fc1 for each token.

    Arguments:

        - experts_for_tokens: [n_tokens, expert_topk] experts assigned to each token.
        - expert_histogram_cumsum: [n_experts]: cumsum of number of tokens allocated to each expert,
          with last element being number of tokens. NOTE: This is usually calculated as part of gemm
          grouped, so you can just reuse it.

    Returns:

        - [n_tokens, expert_topk] row number into activation passed to MoE fc1 for each token. Each
          token should be duplicated `expert_topk` times.
    """
    # No noncontiguous input.
    assert experts_for_tokens.is_contiguous()
    assert experts_for_tokens.numel() < (1 << 31) - 1
    assert expert_histogram_cumsum.is_contiguous()
    assert experts_for_tokens.device == expert_histogram_cumsum.device, (
        f"experts_for_tokens.device = {experts_for_tokens.device}, expert_histogram_cumsum.device = {expert_histogram_cumsum.device}"
    )

    BLOCK_SIZE = 128  # Faster than 1024, not sure why. May be better occupancy?

    histogram_cumsum_copy = expert_histogram_cumsum.clone().detach()  # Temporary workspace.
    indices = torch.empty_like(experts_for_tokens, dtype=int)

    with get_torch_device().device(experts_for_tokens.device):
        _moe_index_compute_kernel[(triton.cdiv(experts_for_tokens.numel(), BLOCK_SIZE),)](
            indices_ptr=indices,
            experts_for_tokens_ptr=experts_for_tokens,
            temp_histogram_cumsum_ptr=histogram_cumsum_copy,
            num_elts=experts_for_tokens.numel(),
            NUM_EXPERTS=histogram_cumsum_copy.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return indices
