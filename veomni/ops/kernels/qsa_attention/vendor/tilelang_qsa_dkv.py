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

"""KV-owned QSA backward with exact reverse adjacency and no floating-point atomics.

KV tiles are storage tiles, not selection blocks: their per-query bit masks
preserve arbitrary global token selections, including packed sample boundaries.
"""

import tilelang
import torch
import triton
import triton.language as tl
from tilelang import language as T


@triton.jit
def _count_tiles(
    Indices, Counts, S: tl.constexpr, K: tl.constexpr, NB: tl.constexpr, BK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    i = tl.program_id(1) * (BLOCK - BK) + lane - BK
    value = tl.load(Indices + row * K + i, (i >= 0) & (i < K), other=-1)
    following = tl.load(Indices + row * K + i + 1, (i + 1 >= 0) & (i + 1 < K), other=-1)
    key = value // BK
    last = (lane >= BK) & (value >= 0) & ((i == K - 1) | (key != following // BK))
    tl.atomic_add(Counts + row // S * NB + key, 1, mask=last, sem="relaxed")


@triton.jit
def _merge_masks(key_a, mask_a, key_b, mask_b):
    return key_b, tl.where(key_a == key_b, mask_a | mask_b, mask_b)


@triton.jit
def _scatter_tiles(
    Indices,
    Cursors,
    Queries,
    Masks,
    S: tl.constexpr,
    K: tl.constexpr,
    NB: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    # Valid token indices are unique, so a sorted KV-tile run has at most BK
    # entries. This lookback preserves runs crossing a metadata chunk boundary.
    i = tl.program_id(1) * (BLOCK - BK) + lane - BK
    value = tl.load(Indices + row * K + i, (i >= 0) & (i < K), other=-1)
    following = tl.load(Indices + row * K + i + 1, (i + 1 >= 0) & (i + 1 < K), other=-1)
    key = value // BK
    bits = tl.where(value >= 0, 1 << (value % BK), 0).to(tl.uint32)
    _, masks = tl.associative_scan((key, bits), 0, _merge_masks)
    last = (lane >= BK) & (value >= 0) & ((i == K - 1) | (key != following // BK))
    position = tl.atomic_add(Cursors + row // S * NB + key, 1, mask=last, sem="relaxed")
    tl.store(Queries + position, row % S, mask=last)
    tl.store(Masks + position, masks, mask=last)


def build_reverse_tiles(indices, kv_len, block_k=16):
    """Build CSR of (query, exact token mask) for each batch/KV storage tile.

    Capacity is bounded by B*S*min(K, ceil(S_kv/block_k)); no data-dependent
    host allocation or dense attention mask is needed. Integer atomics only
    allocate CSR entries, and never accumulate a floating-point gradient.
    """
    batch, seq, topk = indices.shape
    num_blocks = triton.cdiv(kv_len, block_k)
    counts = torch.zeros(batch * num_blocks, dtype=torch.int32, device=indices.device)
    offsets = torch.zeros(batch * num_blocks + 1, dtype=torch.int32, device=indices.device)
    capacity = max(1, batch * seq * min(topk, num_blocks))
    if capacity > torch.iinfo(torch.int32).max:
        raise ValueError("QSA reverse adjacency exceeds the int32 CSR capacity.")
    queries = torch.empty(capacity, dtype=torch.int32, device=indices.device)
    masks = torch.empty(capacity, dtype=torch.int32, device=indices.device)
    if topk:
        sorted_indices = indices.sort(dim=-1).values.contiguous()
        block = 256
        grid = (batch * seq, triton.cdiv(topk, block - block_k))
        _count_tiles[grid](sorted_indices, counts, seq, topk, num_blocks, block_k, block)
        torch.cumsum(counts, dim=0, dtype=torch.int32, out=offsets[1:])
        cursors = offsets[:-1].clone()
        _scatter_tiles[grid](sorted_indices, cursors, queries, masks, seq, topk, num_blocks, block_k, block)
    return offsets, queries, masks


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd_dkv_owned(B, S, S_kv, kv_heads, group_width, D, capacity, splits, sm_scale, block_k=16):
    num_blocks = tilelang.cdiv(S_kv, block_k)
    query_tile = max(1, 32 // group_width)
    block_m = query_tile * group_width
    H = kv_heads * group_width
    shape_q = [B, S, H, D]
    shape_kv = [B, S_kv, kv_heads, D]
    partial_shape = [B, num_blocks, splits, kv_heads, block_k, D]
    scale_log2 = sm_scale * 1.4426950408889634

    @T.prim_func
    def kernel(
        Q: T.Tensor(shape_q, T.bfloat16),
        K: T.Tensor(shape_kv, T.bfloat16),
        V: T.Tensor(shape_kv, T.bfloat16),
        dO: T.Tensor(shape_q, T.bfloat16),
        Lse: T.Tensor([B, S, H], T.float32),
        Delta: T.Tensor([B, S, H], T.float32),
        Offsets: T.Tensor([B * num_blocks + 1], T.int32),
        Queries: T.Tensor([capacity], T.int32),
        Masks: T.Tensor([capacity], T.int32),
        dK: T.Tensor(partial_shape, T.float32),
        dV: T.Tensor(partial_shape, T.float32),
    ):
        with T.Kernel(num_blocks, B * kv_heads, splits, threads=128) as (kb, bh, split):
            batch = bh // kv_heads
            head = bh % kv_heads
            q_shared = T.alloc_shared([block_m, D], T.bfloat16)
            do_shared = T.alloc_shared([block_m, D], T.bfloat16)
            k_shared = T.alloc_shared([block_k, D], T.bfloat16)
            v_shared = T.alloc_shared([block_k, D], T.bfloat16)
            p_shared = T.alloc_shared([block_m, block_k], T.bfloat16)
            ds_shared = T.alloc_shared([block_m, block_k], T.bfloat16)
            query_ids = T.alloc_shared([query_tile], T.int32)
            masks = T.alloc_shared([query_tile], T.int32)
            p = T.alloc_fragment([block_m, block_k], T.float32)
            ds = T.alloc_fragment([block_m, block_k], T.float32)
            dk = T.alloc_fragment([block_k, D], T.float32)
            dv = T.alloc_fragment([block_k, D], T.float32)
            T.clear(dk)
            T.clear(dv)
            for n, d in T.Parallel(block_k, D):
                k_shared[n, d] = T.if_then_else(kb * block_k + n < S_kv, K[batch, kb * block_k + n, head, d], 0)
                v_shared[n, d] = T.if_then_else(kb * block_k + n < S_kv, V[batch, kb * block_k + n, head, d], 0)
            first = Offsets[batch * num_blocks + kb]
            last = Offsets[batch * num_blocks + kb + 1]
            span = T.ceildiv(last - first, splits * query_tile) * query_tile
            begin = first + split * span
            end = T.min(begin + span, last)
            for qi in T.serial(T.ceildiv(T.max(end - begin, 0), query_tile)):
                for j in T.Parallel(query_tile):
                    entry = begin + qi * query_tile + j
                    query_ids[j] = T.if_then_else(entry < end, Queries[entry], 0)
                    masks[j] = T.if_then_else(entry < end, Masks[entry], 0)
                for m, d in T.Parallel(block_m, D):
                    q_shared[m, d] = Q[batch, query_ids[m // group_width], head * group_width + m % group_width, d]
                    do_shared[m, d] = dO[batch, query_ids[m // group_width], head * group_width + m % group_width, d]
                T.gemm(q_shared, k_shared, p, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(do_shared, v_shared, ds, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                for m, n in T.Parallel(block_m, block_k):
                    qid = query_ids[m // group_width]
                    h = head * group_width + m % group_width
                    prob = T.exp2(p[m, n] * scale_log2 - Lse[batch, qid, h])
                    p[m, n] = T.if_then_else((masks[m // group_width] & (1 << n)) != 0, prob, 0)
                    ds[m, n] = p[m, n] * (ds[m, n] - Delta[batch, qid, h]) * sm_scale
                T.copy(p, p_shared)
                T.copy(ds, ds_shared)
                T.gemm(ds_shared, q_shared, dk, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(p_shared, do_shared, dv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)
            T.copy(dk, dK[batch, kb, split, head, :, :])
            T.copy(dv, dV[batch, kb, split, head, :, :])

    return kernel


@triton.jit
def _reduce_partials(
    DK,
    DV,
    OutK,
    OutV,
    S_KV: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    NB: tl.constexpr,
    SPLITS: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token = i // (HKV * D)
    head = i // D % HKV
    dim = i % D
    dk = tl.full((BLOCK,), 0, tl.float32)
    dv = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLITS):
        offset = ((((batch * NB + token // BK) * SPLITS + split) * HKV + head) * BK + token % BK) * D + dim
        dk += tl.load(DK + offset, token < S_KV, other=0)
        dv += tl.load(DV + offset, token < S_KV, other=0)
    tl.store(OutK + batch * S_KV * HKV * D + i, dk, token < S_KV)
    tl.store(OutV + batch * S_KV * HKV * D + i, dv, token < S_KV)


def qsa_dkv_owned(q, k, v, do, indices, lse, delta, sm_scale):
    batch, seq, heads, dim = q.shape
    kv_len, kv_heads = k.shape[1:3]
    block_k = 16
    splits = min(8, max(1, triton.cdiv(seq, 256)))
    offsets, queries, masks = build_reverse_tiles(indices, kv_len, block_k)
    kernel = bwd_dkv_owned(
        batch, seq, kv_len, kv_heads, heads // kv_heads, dim, queries.numel(), splits, sm_scale, block_k
    )
    partial_k, partial_v = kernel(q, k, v, do, lse, delta, offsets, queries, masks)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    _reduce_partials[(triton.cdiv(kv_len * kv_heads * dim, 256), batch)](
        partial_k, partial_v, dk, dv, kv_len, kv_heads, dim, triton.cdiv(kv_len, block_k), splits, block_k, 256
    )
    return dk, dv
