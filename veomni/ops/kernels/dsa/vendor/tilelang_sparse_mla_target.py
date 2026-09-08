# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

# ruff: noqa
"""Head-summed attention probability over the compressed slice, for the DSA
indexer loss (DeepSeek-V3.2 eq. 4).

This consumes the LSE that ``sparse_mqa_fwd_interface`` returns. That LSE is the
*full* CSA denominator: ``sparse_mqa_fwd_interface`` folds the attention sink
into ``sumexp`` before writing it (see ``tilelang_sparse_mla_fwd.py``, "attn_sink:
add exp(attn_sink[h] - max_scaled) to softmax denominator"), and the ``Indices``
it consumed spanned the sliding window as well as the compressed entries. That
is what makes the teacher produced here paper-correct rather than the
compressed-only approximation. If the sink ever moves out of ``sumexp``,
attention stays correct and this silently does not.
"""

import tilelang
import torch
from tilelang import language as T

from veomni.utils.device import get_torch_device


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mqa_target(heads, dim, topk, sm_scale=None, block_I=64, threads=256):
    if dim != tilelang.math.next_power_of_2(dim):
        raise RuntimeError(f"dim must be power of 2, got {dim}")
    if topk % block_I != 0:
        raise RuntimeError(f"topk ({topk}) must be divisible by block_I ({block_I})")
    # One CTA must own every head of a query, or the head sum would need atomics
    # across CTAs. The attention forward replicates over heads above 64; this
    # kernel refuses instead.
    if heads > 64:
        raise RuntimeError(f"target kernel requires heads <= 64 so one block owns the head sum, got {heads}")
    # Admit exactly the set the interface can emit -- it pads to
    # max(next_power_of_2(heads), 16), so {16, 32, 64} -- rather than the weaker
    # `heads % 16 == 0`, which also admits 48. M = 48 is unsound under
    # T.GemmWarpPolicy.FullRow: the partition it picks is 3 warp-rows x 2
    # warp-cols, which does not cover the 8 warps of a 256-thread block, and
    # lowering dies with `Check failed: (m_warp * n_warp == num_warps) is false:
    # m_warp: 3, n_warp: 2, num_warps: 8`. Without this guard that bare TVM
    # InternalError is what a direct caller bypassing the interface would see.
    if heads not in (16, 32, 64):
        raise RuntimeError(f"target kernel requires heads padded to one of (16, 32, 64), got {heads}")
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    target_shape = [batch, seq_len, topk]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H_per_block = heads
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim

    smem_lower_bound = H_per_block * D * 2 + BI * D * 2
    smem_limit = get_torch_device().get_device_properties(None).shared_memory_per_block_optin
    if smem_lower_bound > smem_limit:
        raise RuntimeError(
            f"block_I={block_I}, dim={dim}, heads={heads} needs at least {smem_lower_bound} B "
            f"of shared memory per block, above this device's {smem_limit} B"
        )

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Target: T.Tensor(target_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            mask = T.alloc_fragment([BI], "bool")
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            tgt = T.alloc_fragment([BI], accum_dtype)

            b_i = by
            s_i = bx

            T.copy(Q[b_i, s_i, 0:H_per_block, :D], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=0):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (
                        Indices[b_i, s_i, i_i * BI + bi_i] >= 0 and Indices[b_i, s_i, i_i * BI + bi_i] < seq_len_kv
                    )

                # Same clamped whole-CTA gather as the attention forward: read the
                # row index straight from global memory so layout inference can
                # coalesce the copy. Out-of-range rows are absorbed by the -inf
                # score initialisation below.
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, T.max(T.min(Indices[b_i, s_i, i_i * BI + bi_i], seq_len_kv - 1), 0), d_i
                    ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # P = exp2(scores * sm_scale_log2e - LSE). Mirrors the backward
                # kernel, which is the one that consumes a *final* LSE; the
                # forward's expression works against a running maximum instead.
                # No online maximum is needed here because LSE >= max score by
                # construction, so every exponent is non-positive.
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - Lse[b_i, s_i, h_i])

                # Sum over heads, the non-contiguous axis of acc_s. Routing this
                # through a [BI, H] transpose buffer is not an option: T.copy
                # matches shapes elementwise and never transposes, so the
                # transposed form is rejected outright when H != BI and, far
                # worse, is silently accepted when H == BI -- where it yields the
                # per-head row sum instead of the head sum.
                #
                # `tgt` is reused across the NI iterations, so this relies on
                # T.reduce_sum's default clear=True zeroing it first (the forward
                # passes clear=False explicitly when it wants accumulation). Both
                # that and the moving output slice are covered by the c=128 and
                # c=100 parametrisations of test_target_kernel_matches_reference;
                # at NI == 1 neither can fail.
                T.reduce_sum(acc_s, tgt, dim=0)
                T.copy(tgt, Target[b_i, s_i, i_i * BI : (i_i + 1) * BI])

    return main


def sparse_mqa_target_fwd_interface(q, kv, topk_idxs, lse, sm_scale=None, block_I=64, threads=256):
    """Head-summed, unnormalised attention probability over the compressed slice.

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        topk_idxs: [B, S, C] int32 -- the *compressed* slice only, -1 for misses
        lse:       [B, S, H] fp32 -- from ``sparse_mqa_fwd_interface``
        sm_scale:  float or None (defaults to 1/sqrt(D)); must match the value
                   passed to the forward that produced ``lse``

    Returns:
        [B, S, C] fp32. L1-normalise in the caller.
    """
    # These guard the teacher's numerical contract, so they raise rather than
    # assert: `python -O` strips asserts, and a stripped `heads <= 64` would not
    # fail, it would return a head sum silently missing every head past 64.
    if not (q.is_contiguous() and kv.is_contiguous() and topk_idxs.is_contiguous()):
        raise RuntimeError("target kernel requires q, kv and topk_idxs to be contiguous")
    if not (lse.is_contiguous() and lse.dtype == torch.float32):
        raise RuntimeError(f"target kernel requires a contiguous fp32 lse, got dtype={lse.dtype}")
    # The kernel hardcodes `dtype = T.bfloat16` for both Q and KV, so an fp16 or
    # fp32 caller would otherwise get a tilelang type error from inside lowering
    # instead of a named constraint.
    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"target kernel is bfloat16-only, got q.dtype={q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise RuntimeError(f"target kernel is bfloat16-only, got kv.dtype={kv.dtype}")
    batch, seq_len, heads, dim = q.shape
    if heads > 64:
        raise RuntimeError(f"target kernel requires heads <= 64 so one block owns the head sum, got {heads}")
    if kv.shape[-1] != dim:
        raise RuntimeError(f"kv head dim {kv.shape[-1]} does not match q head dim {dim}")
    # The gather clamps candidate rows into [0, S_kv - 1], which needs a row to
    # exist: with S_kv == 0 the clamp target becomes -1 and the outer max pulls it
    # back to row 0 of an empty tensor, an out-of-bounds device read. The candidate
    # mask only zeroes the score afterwards, so it cannot prevent the gather.
    if kv.shape[1] == 0:
        raise RuntimeError("target kernel requires kv to have at least one row")
    if lse.shape != (batch, seq_len, heads):
        raise RuntimeError(f"lse must be [B, S, H], got {tuple(lse.shape)}")
    topk = topk_idxs.shape[-1]

    padded_topk = (topk + block_I - 1) // block_I * block_I
    if padded_topk != topk:
        pad = torch.full((batch, seq_len, padded_topk - topk), -1, device=topk_idxs.device, dtype=topk_idxs.dtype)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()

    # T.gemm requires the M extent to be a multiple of 16, so head counts below
    # that (or awkward multiples of it) have to be padded, exactly as the
    # attention forward pads. Unlike the forward, the padding here lands *inside*
    # a sum rather than in a slice that can be thrown away afterwards, so a
    # padded head must be inert rather than merely ignorable: zero Q alone would
    # still contribute exp2(0 - lse_pad). Padding the LSE with +inf is what makes
    # the contribution exactly exp2(-inf) == 0. Every score is finite or -inf, so
    # the exponent is never inf - inf and no NaN can enter the head sum.
    padded_heads = max(tilelang.math.next_power_of_2(heads), 16)
    if padded_heads != heads:
        q = torch.cat([q, q.new_zeros(batch, seq_len, padded_heads - heads, dim)], dim=2).contiguous()
        lse = torch.cat([lse, lse.new_full((batch, seq_len, padded_heads - heads), float("inf"))], dim=2).contiguous()

    kernel = sparse_mqa_target(padded_heads, dim, padded_topk, sm_scale, block_I=block_I, threads=threads)
    target = kernel(q, kv, topk_idxs, lse)
    return target[:, :, :topk].contiguous()
