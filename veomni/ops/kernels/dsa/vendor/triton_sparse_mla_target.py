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

"""Triton head-summed attention probability over the compressed slice, for AMD ROCm.

The ROCm counterpart of ``tilelang_sparse_mla_target``, with the same contract:
it consumes the *full* base-2 CSA LSE of the sparse attention forward (sliding
window, compressed entries and sink), so ``exp2(score * scale * log2(e) - lse)``
is each head's share of the layer's joint softmax, and returns the unnormalised
head sum.
"""

import torch
import triton
import triton.language as tl


__all__ = ["sparse_mqa_target_fwd_interface"]

_LOG2E = 1.4426950408889634


@triton.jit
def _target_kernel(
    Q_PTR,
    KV_PTR,
    IDX_PTR,
    LSE_PTR,
    OUT_PTR,
    S,
    P,
    H,
    C,
    SCALE_LOG2E,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = row // S
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H
    q = tl.load(Q_PTR + row * H * D + offs_h[:, None] * D + offs_d[None, :], mask=h_mask[:, None], other=0.0)
    # A padded head gets lse = +inf, so exp2(score - lse) contributes exactly 0.
    lse = tl.load(LSE_PTR + row * H + offs_h, mask=h_mask, other=float("inf"))
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        c_mask = offs_c < C
        idx = tl.load(IDX_PTR + row * C + offs_c, mask=c_mask, other=-1)
        valid = (idx >= 0) & (idx < P)
        kv_rows = b * P + tl.where(valid, idx, 0).to(tl.int64)
        kv = tl.load(KV_PTR + kv_rows[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        probs = tl.exp2(tl.dot(q, tl.trans(kv)) * SCALE_LOG2E - lse[:, None])
        target = tl.sum(tl.where(valid[None, :], probs, 0.0), axis=0)
        tl.store(OUT_PTR + row * C + offs_c, target, mask=c_mask)


def sparse_mqa_target_fwd_interface(q, kv, topk_idxs, lse, sm_scale=None):
    """Head-summed, unnormalised attention probability over the compressed slice.

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        topk_idxs: [B, S, C] int32 -- the *compressed* slice only, -1 for misses
        lse:       [B, S, H] fp32 base-2 LSE of the sparse attention forward
        sm_scale:  float or None (defaults to 1/sqrt(D)); must match the value
                   passed to the forward that produced ``lse``

    Returns:
        [B, S, C] fp32. L1-normalise in the caller.
    """
    if not (q.is_contiguous() and kv.is_contiguous() and topk_idxs.is_contiguous()):
        raise RuntimeError("target kernel requires q, kv and topk_idxs to be contiguous")
    if not (lse.is_contiguous() and lse.dtype == torch.float32):
        raise RuntimeError(f"target kernel requires a contiguous fp32 lse, got dtype={lse.dtype}")
    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"target kernel is bfloat16-only, got q.dtype={q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise RuntimeError(f"target kernel is bfloat16-only, got kv.dtype={kv.dtype}")
    batch, seq_len, heads, dim = q.shape
    if heads > 64:
        raise RuntimeError(f"target kernel requires heads <= 64 so one block owns the head sum, got {heads}")
    if dim < 16 or dim & (dim - 1):
        raise RuntimeError(f"dim must be a power of 2 and at least 16, got {dim}")
    if kv.shape[-1] != dim:
        raise RuntimeError(f"kv head dim {kv.shape[-1]} does not match q head dim {dim}")
    if kv.shape[1] == 0:
        raise RuntimeError("target kernel requires kv to have at least one row")
    if lse.shape != (batch, seq_len, heads):
        raise RuntimeError(f"lse must be [B, S, H], got {tuple(lse.shape)}")
    width = topk_idxs.shape[-1]
    target = torch.empty((batch, seq_len, width), dtype=torch.float32, device=q.device)
    if batch * seq_len and width:
        scale = dim**-0.5 if sm_scale is None else sm_scale
        block_h = max(16, triton.next_power_of_2(heads))
        # Measured on MI308X: a 64-head q tile amortises a 64-slot kv gather, smaller
        # head blocks are latency-bound and run fastest on 16-slot tiles with 2 warps.
        # Software pipelining (num_stages > 1) cost 2.6x at DeepSeek-V4 width.
        block_c, num_warps = (64, 4) if block_h == 64 else (16, 2)
        _target_kernel[(batch * seq_len,)](
            q,
            kv,
            topk_idxs,
            lse,
            target,
            seq_len,
            kv.shape[1],
            heads,
            width,
            scale * _LOG2E,
            D=dim,
            BLOCK_H=block_h,
            BLOCK_C=block_c,
            num_warps=num_warps,
            num_stages=1,
        )
    return target
