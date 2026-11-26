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

"""K32 E4M3/E8M0 grouped GEMMs for merged MoE forward and backward."""

import torch
import triton
import triton.language as tl

from .....utils.device import get_gpu_compute_capability, get_torch_device, is_nvidia_cuda_device


@triton.jit
def _quantize_mx(x, q, s, chunks: tl.constexpr, tile: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64) * (tile // 32) + tl.arange(0, tile // 32)
    k = tl.arange(0, 32)
    idx = chunk[:, None] * 32 + k[None, :]
    val = tl.load(x + idx, chunk[:, None] < chunks, other=0).to(tl.float32)
    peak = tl.max(tl.abs(val), axis=1)
    exponent = tl.ceil(tl.log2(tl.maximum(peak, 2.0**-118) / 448.0))
    exponent = tl.maximum(-126.0, tl.minimum(exponent, 127.0))
    quant = tl.maximum(-448.0, tl.minimum(448.0, val * tl.exp2(-exponent[:, None])))
    tl.store(q + idx, quant.to(tl.float8e4nv).to(tl.uint8, bitcast=True), chunk[:, None] < chunks)
    tl.store(s + chunk, (exponent + 127.0).to(tl.uint8), chunk < chunks)


@triton.jit
def _group_matmul(a, b, sa, sb, c, offsets, M_TILES, N: tl.constexpr, K: tl.constexpr):
    tile = tl.program_id(0)
    expert = tl.program_id(1).to(tl.int64)
    lo = tl.load(offsets + expert - 1, mask=expert > 0, other=0).to(tl.int64)
    hi = tl.load(offsets + expert).to(tl.int64)
    m0 = (tile % M_TILES) * 128
    if lo + m0 >= hi:
        return
    m = lo + m0 + tl.arange(0, 128)
    n = (tile // M_TILES) * 64 + tl.arange(0, 64)
    weight_row = expert * N + n
    k = tl.arange(0, 128)
    scale_k = tl.arange(0, 4)
    acc = tl.zeros((128, 64), dtype=tl.float32)
    for block in range(K // 128):
        ak = block * 128 + k
        sk = block * 4 + scale_k
        av = tl.load(a + m[:, None] * K + ak[None, :], m[:, None] < hi, other=0)
        bv = tl.load(b + weight_row[:, None] * K + ak[None, :], n[:, None] < N, other=0)
        asc = tl.load(sa + m[:, None] * (K // 32) + sk[None, :], m[:, None] < hi, other=127)
        bsc = tl.load(sb + weight_row[:, None] * (K // 32) + sk[None, :], n[:, None] < N, other=127)
        acc = tl.dot_scaled(av, asc, "e4m3", bv.T, bsc, "e4m3", acc)
    tl.store(c + m[:, None] * N + n[None, :], acc, (m[:, None] < hi) & (n[None, :] < N))


def _quantize(matrix):
    q = torch.empty_like(matrix, dtype=torch.uint8)
    chunks = matrix.numel() // 32
    s = torch.empty(chunks, dtype=torch.uint8, device=matrix.device)
    with get_torch_device().device(matrix.device):
        _quantize_mx[(triton.cdiv(chunks, 32),)](matrix, q, s, chunks, 1024)
    return q, s


def _validate_offsets(offsets, tokens, experts, device):
    if (
        offsets.ndim != 1
        or offsets.numel() != experts
        or experts == 0
        or offsets.dtype not in (torch.int32, torch.int64)
        or offsets.device != device
        or not offsets.is_contiguous()
    ):
        raise ValueError("MXFP8 offsets must be contiguous integer [experts] on the operand device")
    # The launch already needs a host count; validate boundaries in the same transfer.
    ends = offsets.tolist()
    counts = [hi - lo for lo, hi in zip([0] + ends[:-1], ends, strict=True)]
    if min(counts) < 0 or ends[-1] != tokens:
        raise ValueError("MXFP8 offsets must be nondecreasing and end at the token count")
    return max(counts), sum((count + 127) // 128 for count in counts)


def _check_device(a, b):
    if not is_nvidia_cuda_device(a.device) or b.device != a.device:
        raise ValueError("MXFP8 operands must be on the same CUDA device")
    if get_gpu_compute_capability(a.device) // 10 != 10:
        raise RuntimeError("MXFP8 requires a Blackwell SM10x CUDA GPU")


def mxfp8_group_gemm(a, b, cumsum_m):
    """Compute grouped A @ B.T with BF16 inputs/output and FP32 accumulation."""
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("MXFP8 expects [tokens, K] and [experts, N, K] operands")
    experts, n, k = b.shape
    if a.dtype != torch.bfloat16 or b.dtype != a.dtype or k == 0 or k % 128 or n == 0:
        raise ValueError("MXFP8 expects BF16 operands, positive N, and positive K divisible by 128")
    if not a.is_contiguous() or not b.is_contiguous() or a.shape[1] != k:
        raise ValueError("MXFP8 expects contiguous [tokens, K] and [experts, N, K] operands")
    _check_device(a, b)
    max_m, _ = _validate_offsets(cumsum_m, a.shape[0], experts, a.device)
    out = a.new_empty((a.shape[0], n))
    if not a.shape[0]:
        return out
    aq, sa = _quantize(a)
    bq, sb = _quantize(b)
    tiles_m = triton.cdiv(max_m, 128)
    with get_torch_device().device(a.device):
        _group_matmul[(tiles_m * triton.cdiv(n, 64), experts)](aq, bq, sa, sb, out, cumsum_m, tiles_m, n, k)
    return out


@triton.jit
def _quantize_wgrad_input(x, q, s, token_ends, block_ends, block_experts, D: tl.constexpr):
    block = tl.program_id(0)
    expert = tl.load(block_experts + block)
    first_block = tl.load(block_ends + expert - 1, mask=expert > 0, other=0)
    first_token = tl.load(token_ends + expert - 1, mask=expert > 0, other=0)
    last_token = tl.load(token_ends + expert)
    token = first_token.to(tl.int64) + (block - first_block).to(tl.int64) * 128 + tl.arange(0, 128)
    feature = tl.program_id(1) * 16 + tl.arange(0, 16)
    valid_feature = feature < D
    value = tl.load(
        x + token[None, :] * D + feature[:, None], valid_feature[:, None] & (token[None, :] < last_token), other=0
    ).to(tl.float32)
    rows = tl.reshape(value, (64, 32))
    peak = tl.max(tl.abs(rows), axis=1)
    exponent = tl.maximum(-126.0, tl.minimum(tl.ceil(tl.log2(tl.maximum(peak, 2.0**-118) / 448.0)), 127.0))
    quant = tl.maximum(-448.0, tl.minimum(448.0, rows * tl.exp2(-exponent[:, None])))
    quant = tl.reshape(quant.to(tl.float8e4nv).to(tl.uint8, bitcast=True), (16, 128))
    base = block.to(tl.int64) * D + feature
    tl.store(q + base[:, None] * 128 + tl.arange(0, 128)[None, :], quant, valid_feature[:, None])
    tl.store(
        s + base[:, None] * 4 + tl.arange(0, 4)[None, :],
        tl.reshape((exponent + 127.0).to(tl.uint8), (16, 4)),
        valid_feature[:, None],
    )


@triton.jit
def _weight_grad(a, b, sa, sb, out, block_ends, M: tl.constexpr, N: tl.constexpr):
    expert = tl.program_id(2).to(tl.int64)
    lo = tl.load(block_ends + expert - 1, mask=expert > 0, other=0)
    hi = tl.load(block_ends + expert)
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 128)
    sk = tl.arange(0, 4)
    acc = tl.zeros((128, 64), dtype=tl.float32)
    for block in range(lo, hi):
        am = block.to(tl.int64) * M + m
        bn = block.to(tl.int64) * N + n
        av = tl.load(a + am[:, None] * 128 + k[None, :], m[:, None] < M, other=0)
        bv = tl.load(b + bn[:, None] * 128 + k[None, :], n[:, None] < N, other=0)
        asc = tl.load(sa + am[:, None] * 4 + sk[None, :], m[:, None] < M, other=127)
        bsc = tl.load(sb + bn[:, None] * 4 + sk[None, :], n[:, None] < N, other=127)
        acc = tl.dot_scaled(av, asc, "e4m3", bv.T, bsc, "e4m3", acc)
    dest = (expert * M + m[:, None]) * N + n[None, :]
    tl.store(out + dest, acc, (m[:, None] < M) & (n[None, :] < N))


def mxfp8_group_wgrad(grad_output, activations, token_ends):
    """Compute each expert's dY.T @ X, resetting scales at expert boundaries."""
    if (
        grad_output.dtype != torch.bfloat16
        or activations.dtype != grad_output.dtype
        or grad_output.ndim != 2
        or activations.ndim != 2
        or grad_output.shape[0] != activations.shape[0]
        or not grad_output.is_contiguous()
        or not activations.is_contiguous()
    ):
        raise ValueError("MXFP8 wgrad expects contiguous BF16 [tokens, features] operands")
    _check_device(grad_output, activations)
    _, total_blocks = _validate_offsets(token_ends, grad_output.shape[0], token_ends.numel(), grad_output.device)
    m, n = grad_output.shape[1], activations.shape[1]
    if m == 0 or n == 0:
        raise ValueError("MXFP8 wgrad feature dimensions must be positive")
    out = grad_output.new_empty((token_ends.numel(), m, n))
    if not grad_output.shape[0]:
        return out.zero_()
    counts = torch.diff(token_ends, prepend=token_ends.new_zeros(1))
    block_counts = (counts + 127) // 128
    block_ends = block_counts.cumsum(0)
    experts = torch.arange(counts.numel(), device=counts.device).repeat_interleave(
        block_counts, output_size=total_blocks
    )

    def pack(x):
        d = x.shape[1]
        q = torch.empty((total_blocks, d, 128), dtype=torch.uint8, device=x.device)
        s = torch.empty((total_blocks, d, 4), dtype=torch.uint8, device=x.device)
        with get_torch_device().device(x.device):
            _quantize_wgrad_input[(total_blocks, triton.cdiv(d, 16))](x, q, s, token_ends, block_ends, experts, d)
        return q, s

    a, sa = pack(grad_output)
    b, sb = pack(activations)
    with get_torch_device().device(grad_output.device):
        _weight_grad[(triton.cdiv(m, 128), triton.cdiv(n, 64), token_ends.numel())](
            a, b, sa, sb, out, block_ends, m, n
        )
    return out
