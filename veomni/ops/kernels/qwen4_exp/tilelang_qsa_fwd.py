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
# Adapted from veomni/ops/kernels/deepseek_v4/tilelang_sparse_mla_fwd.py for
# Qwen4-Exp QSA.
# Key differences from DeepSeek-V4:
#   - GQA: K and V are separate tensors with ``kv_heads`` heads. Each CTA owns
#     the query-head group of exactly one KV head, so the gathered K/V tile is
#     shared by every query head in the CTA.
#   - No attention sink: the softmax denominator comes from selected tokens
#     only.
#   - An all-(-1) query row yields exactly-zero outputs and a zero LSE, so the
#     backward kernel never sees exp2(-inf - (-inf)).
import tilelang
import torch
from tilelang import language as T

from ....utils.device import get_torch_device


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def qsa_fwd(
    kv_heads,
    group_width,
    dim,
    topk,
    sm_scale=None,
    block_I=64,
    num_stages=0,
    threads=256,
):
    # ``group_width`` is the (padded) number of query heads per KV head, i.e. the
    # GQA group size. It is the tile M dimension, so it must be a power of two
    # between 16 and 64; the interface pads real groups up to it.
    #
    # ``num_stages >= 1`` pipelines the KV gather, and that is only legal because the
    # gather takes its row index straight from ``Indices`` in global memory. See the
    # DeepSeek-V4 forward kernel for why a register-fragment index breaks the
    # pipeliner. It stays off by default here as it did there: re-enable only on an
    # end-to-end measurement, never on a microbenchmark.
    assert num_stages >= 0, f"num_stages must be non-negative, got {num_stages}"
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be power of 2, got {dim}"
    assert topk % block_I == 0, f"topk ({topk}) must be divisible by block_I ({block_I})"
    assert group_width == tilelang.math.next_power_of_2(group_width), (
        f"group_width must be power of 2, got {group_width}"
    )
    assert 16 <= group_width <= 64, f"group_width must be in [16, 64], got {group_width}"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    heads = kv_heads * group_width
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, kv_heads, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = group_width
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim

    # Every stage past the first double-buffers the BI x D K and V tiles, so the
    # depth is bounded by shared memory rather than by anything in the algorithm.
    smem_lower_bound = (
        G * D * 2  # Q_shared, bf16
        + G * BI * 2  # S_shared, bf16
        + 2 * max(num_stages, 1) * BI * D * 2  # K_shared + V_shared, one buffer per stage each
    )
    # Every kernel here is CUDA-only; the device-agnostic helper stays because
    # `device-api-check` rejects a vendor-namespaced device reference under
    # veomni/ when veomni/utils/device.py has an equivalent, and here it does.
    smem_limit = get_torch_device().get_device_properties(None).shared_memory_per_block_optin
    assert smem_lower_bound <= smem_limit, (
        f"num_stages={num_stages} at block_I={block_I}, dim={dim} needs at least "
        f"{smem_lower_bound} B of shared memory per block, above this device's {smem_limit} B"
    )

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, kv_heads, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, D], dtype)
            K_shared = T.alloc_shared([BI, D], dtype)
            V_shared = T.alloc_shared([BI, D], dtype)
            mask = T.alloc_fragment([BI], "bool")
            mask_shared = T.alloc_shared([BI], "bool")

            acc_o = T.alloc_fragment([G, D], accum_dtype)
            acc_s = T.alloc_fragment([G, BI], accum_dtype)
            S_shared = T.alloc_shared([G, BI], dtype)
            sumexp = T.alloc_fragment([G], accum_dtype)
            sumexp_i = T.alloc_fragment([G], accum_dtype)
            alpha = T.alloc_fragment([G], accum_dtype)
            m_i = T.alloc_fragment([G], accum_dtype)
            m_i_prev = T.alloc_fragment([G], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i = by
            s_i = bx
            # Query heads of KV head ``bz`` occupy the contiguous slice
            # [bz * G, (bz + 1) * G), matching Transformers' repeat_kv grouping.
            H0 = bz * G

            T.copy(Q[b_i, s_i, H0 : H0 + G, :D], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (
                        Indices[b_i, s_i, i_i * BI + bi_i] >= 0 and Indices[b_i, s_i, i_i * BI + bi_i] < seq_len_kv
                    )
                # Stage the mask through shared memory: reading the mask *fragment*
                # inside the [G, BI] pre-set loop below forces a per-tile
                # cross-thread layout conversion of the fragment (measured: ~2.7x
                # slowdown of the equivalent backward kernel). A shared read
                # broadcasts per bank instead.
                T.copy(mask, mask_shared)

                # Read the row index straight from global memory and clamp it, instead of
                # going through a register fragment; see the DeepSeek-V4 kernel for why
                # the fragment form collapses the copy onto a handful of threads and
                # breaks pipelining. Clamping is equivalent to substituting row 0:
                # out-of-range candidates have their scores pre-set to -inf below, which
                # absorbs the finite QK product from whichever real row is fetched here,
                # so their softmax weight is exactly zero and they contribute nothing to
                # the PV GEMM.
                for bi_i, d_i in T.Parallel(BI, D):
                    K_shared[bi_i, d_i] = K[
                        b_i, T.max(T.min(Indices[b_i, s_i, i_i * BI + bi_i], seq_len_kv - 1), 0), bz, d_i
                    ]
                for bi_i, d_i in T.Parallel(BI, D):
                    V_shared[bi_i, d_i] = V[
                        b_i, T.max(T.min(Indices[b_i, s_i, i_i * BI + bi_i], seq_len_kv - 1), 0), bz, d_i
                    ]

                for h_i, bi_i in T.Parallel(G, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask_shared[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(G):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(G):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(G, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(G):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(G, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Empty-row guard: a query whose selection is entirely -1 has sumexp == 0.
            # Emit exactly zero and a zero LSE; the backward kernel then computes
            # P = exp2(-inf - 0) = 0 for every masked entry instead of exp2(nan).
            for h_i, d_i in T.Parallel(G, D):
                acc_o[h_i, d_i] = T.if_then_else(sumexp[h_i] > 0, acc_o[h_i, d_i] / sumexp[h_i], 0.0)
            # LSE = log2(sumexp) + m_i * sm_scale (in log2 space)
            for h_i in T.Parallel(G):
                sumexp[h_i] = T.if_then_else(sumexp[h_i] > 0, T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale, 0.0)

            T.copy(acc_o, Output[b_i, s_i, H0 : H0 + G, :])
            T.copy(sumexp, Lse[b_i, s_i, H0 : H0 + G])

    return main


def _pad_head_dim(t: torch.Tensor, padded_dim: int) -> torch.Tensor:
    """Zero-pad the trailing head-dim axis of a ``[B, S, H, D]`` tensor.

    The padding is exact: zero K dims leave the QK^T scores unchanged, and
    zero V/dO dims produce zero output/gradient columns that the caller
    slices off after the kernel.
    """
    dim = t.shape[-1]
    if padded_dim == dim:
        return t
    return torch.nn.functional.pad(t, (0, padded_dim - dim))


def _pad_query_heads(t: torch.Tensor, kv_heads: int, padded_group: int) -> torch.Tensor:
    """Pad the GQA group dimension of a ``[B, S, H, ...]`` tensor with zeros.

    Query heads are laid out as ``kv_heads`` contiguous groups (Transformers
    ``repeat_kv`` order), so padding appends zero heads inside each group rather
    than at the end of the head axis.
    """
    batch, seq_len, heads = t.shape[:3]
    tail = t.shape[3:]
    group = heads // kv_heads
    if padded_group == group:
        return t
    t = t.view(batch, seq_len, kv_heads, group, *tail)
    t = torch.nn.functional.pad(t, (0, 0) * len(tail) + (0, padded_group - group))
    return t.reshape(batch, seq_len, kv_heads * padded_group, *tail).contiguous()


def _slice_query_heads(t: torch.Tensor, kv_heads: int, group: int) -> torch.Tensor:
    """Inverse of ``_pad_query_heads``: drop the padded heads inside each group."""
    batch, seq_len, heads = t.shape[:3]
    tail = t.shape[3:]
    padded_group = heads // kv_heads
    if padded_group == group:
        return t.contiguous()
    t = t.view(batch, seq_len, kv_heads, padded_group, *tail)[:, :, :, :group]
    return t.reshape(batch, seq_len, kv_heads * group, *tail).contiguous()


def qsa_fwd_interface(q, k, v, selected_indices, sm_scale=None, block_I=64, num_stages=0, threads=256):
    """Forward interface for Qwen4-Exp sparse GQA attention.

    Args:
        q:                [B, S, H, D] bf16
        k:                [B, S_kv, H_kv, D] bf16
        v:                [B, S_kv, H_kv, D] bf16
        selected_indices: [B, S, topk] int32, -1 padding
        sm_scale:         float or None (defaults to 1/sqrt(D))

    Returns:
        out: [B, S, H, D] bf16
        lse: [B, S, H] fp32 (log2-space log-sum-exp of the scaled scores)
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert selected_indices.is_contiguous()
    batch, seq_len, heads, dim = q.shape
    _, seq_len_kv, kv_heads, kv_dim = k.shape
    assert v.shape == k.shape, f"k/v shapes must match; got k={k.shape}, v={v.shape}"
    assert kv_dim == dim
    # The gather clamps candidate rows into [0, seq_len_kv - 1], which needs a row to exist.
    assert seq_len_kv > 0, "kv must have at least one row"
    assert heads % kv_heads == 0, f"query heads ({heads}) must be divisible by kv heads ({kv_heads})"
    _, _, topk = selected_indices.shape

    group_width = heads // kv_heads
    padded_group = max(tilelang.math.next_power_of_2(group_width), 16)
    if padded_group > 64:
        raise ValueError(f"QSA TileLang forward supports at most 64 query heads per KV head; got group={group_width}.")
    q = _pad_query_heads(q, kv_heads, padded_group)

    # TileLang's MMA warp tiling floors the PV GEMM's N tile at 64 (8 warps x
    # 8 columns per warp), so a smaller head dim (e.g. the toy config's 16)
    # fails layout inference; zero-padding D to 64 is exact (see _pad_head_dim).
    padded_dim = max(64, dim)
    q = _pad_head_dim(q, padded_dim)
    k = _pad_head_dim(k, padded_dim)
    v = _pad_head_dim(v, padded_dim)

    # Pad topk to next multiple of block_I (kernel requires divisibility)
    padded_topk = (topk + block_I - 1) // block_I * block_I
    if padded_topk != topk:
        pad = torch.full(
            (batch, seq_len, padded_topk - topk), -1, device=selected_indices.device, dtype=selected_indices.dtype
        )
        selected_indices = torch.cat([selected_indices, pad], dim=-1).contiguous()
        topk = padded_topk

    kernel = qsa_fwd(
        kv_heads,
        padded_group,
        padded_dim,
        topk,
        sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, k, v, selected_indices)
    return (
        _slice_query_heads(out, kv_heads, group_width)[..., :dim].contiguous(),
        _slice_query_heads(lse, kv_heads, group_width),
    )
