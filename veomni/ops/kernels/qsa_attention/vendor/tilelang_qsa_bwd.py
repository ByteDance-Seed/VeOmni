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
# Adapted from radixark/miles via veomni/ops/kernels/deepseek_v4; modified for VeOmni.

# ruff: noqa
# Adapted from veomni/ops/kernels/deepseek_v4/tilelang_sparse_mla_bwd.py for
# Qwen4-Exp QSA.
# dQ retains the query-owned sparse gather kernel. dK/dV are reduced by
# KV-owned CTAs in tilelang_qsa_dkv.py, using exact reverse adjacency and
# disjoint FP32 partials instead of floating-point atomic scatter.
import tilelang
import torch
from tilelang import language as T

from .tilelang_qsa_dkv import qsa_dkv_owned


def cta_threads(block_H, block_size):
    """Pick the backward CTA width for a ``(block_H, block_size)`` tile.

    The accumulator fragments are spread over the CTA's threads, so this sets their
    per-thread register footprint. A wider CTA halves every fragment and trades
    ptxas spills for more resident warps.

    The ceiling is the GEMM tiling. ``FullCol`` splits the ``block_size`` columns
    over at most ``block_size // 8`` warps (the MMA n-tile is 8) and the remaining
    warps divide ``block_H`` rows, so each warp's row tile is
    ``block_H * (block_size // 8) / num_warps`` and a warp needs a whole 16-row MMA
    tile. That bounds the width at ``block_size * block_H // 4``; exceeding it is a
    hard TileLang assertion ("warp_row_tiles must be greater than 16"), not a slow
    kernel.
    """
    return min(256, block_size * block_H // 4)


@tilelang.jit(out_idx=[-1])
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd_dq(
    B,
    S,
    S_kv,
    kv_heads,
    group_width,
    D,
    topk,
    sm_scale=None,
    block_size=64,
    num_stages=0,
    threads=None,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """dQ kernel: dQ = (P * (dO @ V^T - Delta) * sm_scale) @ K over selected rows."""
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H = kv_heads * group_width
    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, kv_heads, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    block_H = min(64, group_width)
    assert group_width % block_H == 0
    NH = group_width // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    if threads is None:
        threads = cta_threads(block_H, BS)
    assert threads % 32 == 0 and threads & (threads - 1) == 0, (
        f"threads ({threads}) must be a power-of-two multiple of the 32-lane warp"
    )
    assert threads <= BS * block_H // 4, (
        f"threads ({threads}) exceeds the GEMM warp-tile bound {BS * block_H // 4} "
        f"for block_H={block_H}, block_size={BS}"
    )

    @T.prim_func
    def qsa_bwd_dq_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(S, B, kv_heads * NH, threads=threads) as (s_i, by, bz):
            kv_head = bz // NH
            h0 = kv_head * group_width + (bz % NH) * block_H

            Q_shared = T.alloc_shared([block_H, D], dtype)
            K_shared = T.alloc_shared([BS, D], dtype)
            V_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")
            mask_shared = T.alloc_shared([BS], "bool")

            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)

            T.copy(Q[by, s_i, h0 : h0 + block_H, :D], Q_shared)
            T.copy(dO[by, s_i, h0 : h0 + block_H, :D], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] >= 0 and Indices[by, s_i, i_i * BS + bi_i] < S_kv
                # Stage the mask through shared memory: reading the mask *fragment*
                # inside the [block_H, BS] pre-set loop below forces a per-tile
                # cross-thread layout conversion of the fragment. A shared read
                # broadcasts per bank instead.
                T.copy(mask, mask_shared)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask_shared[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    K_shared[bi_i, d_i] = K[
                        by, T.max(T.min(Indices[by, s_i, i_i * BS + bi_i], S_kv - 1), 0), kv_head, d_i
                    ]
                for bi_i, d_i in T.Parallel(BS, D):
                    V_shared[bi_i, d_i] = V[
                        by, T.max(T.min(Indices[by, s_i, i_i * BS + bi_i], S_kv - 1), 0), kv_head, d_i
                    ]

                T.gemm(Q_shared, K_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                # P = exp2(scores * sm_scale_log2e - LSE); zero for masked entries
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, h0 + h_i])

                # dS = P * (dO @ V^T - Delta) * sm_scale
                T.gemm(
                    dO_shared, V_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, h0 + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)

                # dQ += dS @ K
                T.gemm(dP_shared_cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

            # Store dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, h0 : h0 + block_H, :D])

    return qsa_bwd_dq_kernel


def qsa_bwd_interface(q, k, v, o, do, selected_indices, lse, sm_scale=None):
    """Backward interface for Qwen4-Exp sparse GQA attention.

    Args:
        q:                [B, S, H, D] bf16
        k:                [B, S_kv, H_kv, D] bf16
        v:                [B, S_kv, H_kv, D] bf16
        o:                [B, S, H, D] bf16 (forward output)
        do:               [B, S, H, D] bf16 (grad of output)
        selected_indices: [B, S, topk] int32
        lse:              [B, S, H] fp32 (log2-space log-sum-exp from forward)
        sm_scale:         float or None

    Returns:
        dq: [B, S, H, D] bf16
        dk: [B, S_kv, H_kv, D] bf16
        dv: [B, S_kv, H_kv, D] bf16
    """
    from .tilelang_qsa_fwd import _pad_head_dim, _pad_query_heads, _slice_query_heads

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert o.is_contiguous() and do.is_contiguous()
    assert selected_indices.is_contiguous() and lse.is_contiguous()
    B, S, H, D = q.shape
    if sm_scale is None:
        sm_scale = D**-0.5
    S_kv, kv_heads = k.shape[1], k.shape[2]
    assert v.shape == k.shape, f"k/v shapes must match; got k={k.shape}, v={v.shape}"
    # The gather clamps candidate rows into [0, S_kv - 1], which needs a row to exist.
    assert S_kv > 0, "kv must have at least one row"
    assert H % kv_heads == 0, f"query heads ({H}) must be divisible by kv heads ({kv_heads})"
    topk = selected_indices.shape[-1]

    group_width = H // kv_heads
    padded_group = max(tilelang.math.next_power_of_2(group_width), 16)
    if padded_group > 64:
        raise ValueError(
            f"QSA TileLang backward supports at most 64 query heads per KV head; got group={group_width}."
        )
    q = _pad_query_heads(q, kv_heads, padded_group)
    o = _pad_query_heads(o, kv_heads, padded_group)
    do = _pad_query_heads(do, kv_heads, padded_group)
    lse = _pad_query_heads(lse, kv_heads, padded_group)

    # Same MMA tiling floor as the forward (the preprocess tiles D by 32 and
    # the GEMMs need a >=64-wide N tile); zero-padding D to 64 is exact.
    padded_dim = max(64, D)
    q = _pad_head_dim(q, padded_dim)
    k = _pad_head_dim(k, padded_dim)
    v = _pad_head_dim(v, padded_dim)
    o = _pad_head_dim(o, padded_dim)
    do = _pad_head_dim(do, padded_dim)

    # Pad topk to next multiple of block_size (kernel requires divisibility)
    block_size = 64
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = torch.full((B, S, padded_topk - topk), -1, device=selected_indices.device, dtype=selected_indices.dtype)
        selected_indices = torch.cat([selected_indices, pad], dim=-1).contiguous()
        topk = padded_topk

    preprocess_kernel = preprocess(B, S, kv_heads * padded_group, padded_dim)
    dq_kernel = bwd_dq(B, S, S_kv, kv_heads, padded_group, padded_dim, topk, sm_scale)

    delta = preprocess_kernel(o, do)
    dq = dq_kernel(q, k, v, do, selected_indices, lse, delta)
    dk, dv = qsa_dkv_owned(q, k, v, do, selected_indices, lse, delta, sm_scale)
    return (
        _slice_query_heads(dq, kv_heads, group_width)[..., :D].contiguous(),
        dk[..., :D].contiguous(),
        dv[..., :D].contiguous(),
    )
