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

"""Triton DeepSeek-V4 lightning indexer for AMD ROCm.

Same contract as the TileLang indexer: one fused kernel takes each query's
per-head ``q @ k^T`` against a pool tile with FP32 accumulators, applies ReLU,
the per-head weight, the head sum and the query's visible ``[ks, ke)`` range,
and skips pool tiles the query cannot see. Pool-column chunks feed a running
``torch.topk``, so the full ``[B, S, P]`` row is never materialized. Only the
selected scores are differentiable, so the backward recomputes their logits.
Neither direction materializes ``[B, S, H, P]``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..topk import mask_topk_indices_to_query_range


__all__ = ["triton_lighting_indexer"]

# Each scores program reuses one 128-entry pool tile for 16 queries (V4 widths on
# gfx942, S=4096: 0.61 ms against 1.84 ms for a BF16 GEMM plus a separate tail).
_SCORES_BLOCK_S = 16
_SCORES_BLOCK_P = 128
_SCORES_NUM_WARPS = 4
# Bound the transient FP32 score chunk. Production through roughly 64K tokens
# stays one-shot; longer contexts stream pool columns through a running exact
# top-k. At S=128K, 4 GiB cuts the measured peak from 18 to 11 GiB while keeping
# fwd+bwd within 9% of one-shot. This matches #1271's backward workspace cap.
_SCORES_CHUNK_BYTES = 4 << 30
_MAX_POOL_CHUNK = 32768
# One unpipelined wavefront per query with 16-entry tiles hides the dK atomics best on
# gfx942 (V4 widths, S=4096: 3.0 ms against 6.7 ms for 32-entry tiles on four wavefronts).
_BWD_BLOCK_K = 16
_BWD_NUM_WARPS = 1
_BWD_NUM_STAGES = 1


@triton.jit
def _scores_kernel(
    Q_PTR,
    K_PTR,
    W_PTR,
    KS_PTR,
    KE_PTR,
    OUT_PTR,
    S,
    P,
    P_START,
    P_CHUNK,
    H,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    s_blocks = tl.cdiv(S, BLOCK_S)
    b = (tl.program_id(0) // s_blocks).to(tl.int64)
    s0 = (tl.program_id(0) % s_blocks) * BLOCK_S
    p0 = tl.program_id(1) * BLOCK_P
    offs_local_p = p0 + tl.arange(0, BLOCK_P)
    global_p0 = P_START + p0
    offs_p = P_START + offs_local_p
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    p_mask = (offs_local_p < P_CHUNK) & (offs_p < P)
    h_mask = offs_h < H
    k = tl.load(K_PTR + (b * P + offs_p[:, None]) * D + offs_d[None, :], mask=p_mask[:, None], other=0.0)
    for i in range(BLOCK_S):
        s = s0 + i
        if s < S:
            row = b * S + s
            ks = tl.load(KS_PTR + s)
            ke = tl.load(KE_PTR + s)
            if (global_p0 < ke) & (global_p0 + BLOCK_P > ks):
                q = tl.load(
                    Q_PTR + row * H * D + offs_h[:, None] * D + offs_d[None, :], mask=h_mask[:, None], other=0.0
                )
                w = tl.load(W_PTR + row * H + offs_h, mask=h_mask, other=0.0)
                score = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)
                visible = (offs_p >= ks) & (offs_p < ke)
                tl.store(
                    OUT_PTR + row * P_CHUNK + offs_local_p,
                    tl.where(visible, score, float("-inf")),
                    mask=p_mask,
                )
            else:
                tl.store(
                    OUT_PTR + row * P_CHUNK + offs_local_p,
                    tl.full((BLOCK_P,), float("-inf"), tl.float32),
                    mask=p_mask,
                )


@triton.jit
def _selected_scores_kernel(
    Q_PTR,
    K_PTR,
    W_PTR,
    IDX_PTR,
    KS_PTR,
    KE_PTR,
    OUT_PTR,
    S,
    P,
    H,
    TOPK,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Score caller-provided candidates without materializing the full pool."""
    row = tl.program_id(0).to(tl.int64)
    b = row // S
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H
    q = tl.load(Q_PTR + row * H * D + offs_h[:, None] * D + offs_d[None, :], mask=h_mask[:, None], other=0.0)
    w = tl.load(W_PTR + row * H + offs_h, mask=h_mask, other=0.0)
    ks = tl.load(KS_PTR + row % S)
    ke = tl.load(KE_PTR + row % S)
    for k0 in range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < TOPK
        idx = tl.load(IDX_PTR + row * TOPK + offs_k, mask=k_mask, other=-1)
        valid = k_mask & (idx >= 0) & (idx < P) & (idx >= ks) & (idx < ke)
        kv_rows = b * P + tl.where(valid, idx, 0).to(tl.int64)
        kk = tl.load(K_PTR + kv_rows[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        logits = tl.dot(kk, tl.trans(q))
        scores = tl.sum(tl.maximum(logits, 0.0) * w[None, :], axis=1)
        tl.store(OUT_PTR + row * TOPK + offs_k, tl.where(valid, scores, float("-inf")), mask=k_mask)


@triton.jit
def _sparse_bwd_kernel(
    Q_PTR,
    K_PTR,
    W_PTR,
    IDX_PTR,
    G_PTR,
    DQ_PTR,
    DW_PTR,
    DK_PTR,
    S,
    P,
    H,
    TOPK,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = row // S
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H
    q = tl.load(Q_PTR + row * H * D + offs_h[:, None] * D + offs_d[None, :], mask=h_mask[:, None], other=0.0)
    w = tl.load(W_PTR + row * H + offs_h, mask=h_mask, other=0.0)
    dq = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    dw = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k0 in range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < TOPK
        idx = tl.load(IDX_PTR + row * TOPK + offs_k, mask=k_mask, other=-1)
        valid = (idx >= 0) & (idx < P)
        g = tl.where(valid, tl.load(G_PTR + row * TOPK + offs_k, mask=k_mask, other=0.0), 0.0)
        kv_rows = b * P + tl.where(valid, idx, 0).to(tl.int64)
        kk = tl.load(K_PTR + kv_rows[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0)
        logits = tl.dot(kk, tl.trans(q))
        dw += tl.sum(tl.maximum(logits, 0.0) * g[:, None], axis=0)
        grad_logits = tl.where(logits > 0.0, g[:, None] * w[None, :], 0.0).to(kk.dtype)
        dq += tl.dot(tl.trans(grad_logits), kk)
        tl.atomic_add(
            DK_PTR + kv_rows[:, None] * D + offs_d[None, :],
            tl.dot(grad_logits, q),
            mask=valid[:, None],
            sem="relaxed",
        )
    tl.store(
        DQ_PTR + row * H * D + offs_h[:, None] * D + offs_d[None, :],
        dq.to(DQ_PTR.dtype.element_ty),
        mask=h_mask[:, None],
    )
    tl.store(DW_PTR + row * H + offs_h, dw, mask=h_mask)


def _pool_chunk_width(batch: int, seq_len: int, pool: int) -> int:
    """Pool columns that keep one FP32 score chunk inside the memory budget."""
    if pool <= 0:
        return 0
    bytes_per_column = max(1, batch * seq_len * torch.float32.itemsize)
    by_budget = max(1, _SCORES_CHUNK_BYTES // bytes_per_column)
    chunk = min(pool, _MAX_POOL_CHUNK, by_budget)
    if chunk >= _SCORES_BLOCK_P:
        chunk = max(_SCORES_BLOCK_P, chunk // _SCORES_BLOCK_P * _SCORES_BLOCK_P)
    return chunk


def _pool_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    pool_start: int,
    pool_end: int,
) -> torch.Tensor:
    """FP32 scores for global pool columns ``[pool_start, pool_end)``."""
    batch, seq_len, heads, dim = q.shape
    pool = k.shape[1]
    width = pool_end - pool_start
    scores = torch.empty((batch, seq_len, width), dtype=torch.float32, device=q.device)
    _scores_kernel[(batch * triton.cdiv(seq_len, _SCORES_BLOCK_S), triton.cdiv(width, _SCORES_BLOCK_P))](
        q,
        k,
        w,
        ks,
        ke,
        scores,
        seq_len,
        pool,
        pool_start,
        width,
        heads,
        D=dim,
        BLOCK_H=max(16, triton.next_power_of_2(heads)),
        BLOCK_S=_SCORES_BLOCK_S,
        BLOCK_P=_SCORES_BLOCK_P,
        num_warps=_SCORES_NUM_WARPS,
        num_stages=1,
    )
    return scores


def _streaming_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k over pool-column chunks; ties may choose equivalent columns."""
    batch, seq_len, _, _ = q.shape
    pool = k.shape[1]
    width = min(topk, pool)
    if width == 0:
        shape = (batch, seq_len, 0)
        return q.new_empty(shape, dtype=torch.float32), torch.empty(shape, dtype=torch.int32, device=q.device)

    chunk = _pool_chunk_width(batch, seq_len, pool)
    running_values = running_indices = None
    for pool_start in range(0, pool, chunk):
        pool_end = min(pool_start + chunk, pool)
        scores = _pool_scores(q, k, w, ks, ke, pool_start, pool_end)
        values, local_indices = scores.topk(min(width, pool_end - pool_start), dim=-1)
        indices = local_indices + pool_start
        indices = torch.where(torch.isfinite(values), indices, -1).to(torch.int32)
        if running_values is None:
            running_values, running_indices = values, indices
            continue
        candidate_values = torch.cat((running_values, values), dim=-1)
        candidate_indices = torch.cat((running_indices, indices), dim=-1)
        running_values, selected = candidate_values.topk(min(width, candidate_values.shape[-1]), dim=-1)
        running_indices = torch.gather(candidate_indices, -1, selected)

    running_indices = torch.where(torch.isfinite(running_values), running_indices, -1)
    return running_values, running_indices


def _selected_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Score an existing ``[B, S, K]`` selection directly."""
    batch, seq_len, heads, dim = q.shape
    pool = k.shape[1]
    width = indices.shape[-1]
    scores = torch.empty(indices.shape, dtype=torch.float32, device=q.device)
    if batch * seq_len and width:
        _selected_scores_kernel[(batch * seq_len,)](
            q,
            k,
            w,
            indices.contiguous(),
            ks,
            ke,
            scores,
            seq_len,
            pool,
            heads,
            width,
            D=dim,
            BLOCK_H=max(16, triton.next_power_of_2(heads)),
            BLOCK_K=_BWD_BLOCK_K,
            num_warps=_BWD_NUM_WARPS,
            num_stages=_BWD_NUM_STAGES,
        )
    return scores


class _TritonIndexer(torch.autograd.Function):
    """``q [B, S, H, D]`` BF16, ``k [B, P, D]`` BF16, ``w [B, S, H]`` FP32, ranges ``[S]`` int32."""

    @staticmethod
    def forward(ctx, q, k, w, ks, ke, topk: int, topk_indices):
        batch, seq_len, _, _ = q.shape
        pool = k.shape[1]
        if pool == 0:
            width = 0 if topk_indices is None else topk_indices.shape[-1]
            indices = (
                torch.full((batch, seq_len, width), -1, dtype=torch.int32, device=q.device)
                if topk_indices is None
                else topk_indices
            )
            ctx.save_for_backward(q, k, w, indices.new_full(indices.shape, -1))
            ctx.mark_non_differentiable(indices)
            return q.new_full((batch, seq_len, width), float("-inf"), dtype=torch.float32), indices

        if topk_indices is None:
            selected, indices = _streaming_topk(q, k, w, ks, ke, topk)
        else:
            indices = topk_indices
            selected = _selected_scores(q, k, w, ks, ke, indices)
        visible_indices, _ = mask_topk_indices_to_query_range(indices.to(torch.int32), ks, ke)
        visible_indices = visible_indices.masked_fill(visible_indices >= pool, -1)
        ctx.save_for_backward(q, k, w, visible_indices.contiguous())
        ctx.mark_non_differentiable(indices)
        return selected, indices

    @staticmethod
    def backward(ctx, grad_scores, _grad_indices):
        q, k, w, indices = ctx.saved_tensors
        batch, seq_len, heads, dim = q.shape
        pool = k.shape[1]
        dq = torch.zeros_like(q)
        dw = torch.zeros_like(w)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        width = indices.shape[-1]
        if pool and width and batch * seq_len:
            _sparse_bwd_kernel[(batch * seq_len,)](
                q,
                k,
                w,
                indices,
                grad_scores.float().contiguous(),
                dq,
                dw,
                dk,
                seq_len,
                pool,
                heads,
                width,
                D=dim,
                BLOCK_H=max(16, triton.next_power_of_2(heads)),
                BLOCK_K=_BWD_BLOCK_K,
                num_warps=_BWD_NUM_WARPS,
                num_stages=_BWD_NUM_STAGES,
            )
        return dq, dk.to(k.dtype), dw, None, None, None, None


def triton_lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: torch.Tensor | None = None,
    cu_seqlen_ks: torch.Tensor | None = None,
    cu_seqlen_ke: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score and select compressed KV entries.

    Args:
        index_q: ``[S, B, H, D]`` BF16.
        index_k: ``[S_kv, B, D]`` BF16.
        weights: ``[S, B, H]`` FP32, including the softmax scale.
        compress_ratio: Compression ratio; without packed ranges, query ``t``
            sees entries ``[0, (t + 1) // compress_ratio)``.
        topk: Number of entries to select.
        topk_indices: Optional ``[B, S, K]`` int32 selection to score instead.
        cu_seqlen_ks / cu_seqlen_ke: Optional ``[S]`` int32 visible ranges.

    Returns:
        ``(scores, indices)``, both ``[B, S, K]``: FP32 scores (``-inf`` for
        unusable slots) and int32 indices (``-1`` for unusable slots).
    """
    if index_q.ndim != 4:
        raise ValueError(f"DeepSeek V4 Triton indexer requires q [S, B, H, D], got {tuple(index_q.shape)}")
    if index_k.ndim != 3:
        raise ValueError(f"DeepSeek V4 Triton indexer requires k [P, B, D], got {tuple(index_k.shape)}")
    if weights.ndim != 3:
        raise ValueError(f"DeepSeek V4 Triton indexer requires weights [S, B, H], got {tuple(weights.shape)}")

    seq_len, batch, heads, dim = index_q.shape
    pool, key_batch, key_dim = index_k.shape
    if key_batch != batch or key_dim != dim:
        raise ValueError(f"DeepSeek V4 Triton indexer requires k [P, B={batch}, D={dim}], got {tuple(index_k.shape)}")
    if weights.shape != (seq_len, batch, heads):
        raise ValueError(
            f"DeepSeek V4 Triton indexer requires weights [S={seq_len}, B={batch}, H={heads}], "
            f"got {tuple(weights.shape)}"
        )
    if index_q.device != index_k.device or index_q.device != weights.device:
        raise ValueError(
            "DeepSeek V4 Triton indexer requires q, k and weights on one device, got "
            f"q={index_q.device}, k={index_k.device}, weights={weights.device}"
        )
    if not isinstance(compress_ratio, int) or compress_ratio <= 0:
        raise ValueError(f"DeepSeek V4 Triton indexer requires compress_ratio > 0, got {compress_ratio!r}")
    if not isinstance(topk, int) or topk < 0:
        raise ValueError(f"DeepSeek V4 Triton indexer requires topk >= 0, got {topk!r}")
    if heads > 64:
        raise ValueError(f"DeepSeek V4 Triton indexer supports at most 64 heads, got {heads}")
    if dim < 16 or dim & (dim - 1):
        raise ValueError(f"DeepSeek V4 Triton indexer requires a power-of-two head dim >= 16, got {dim}")
    if index_q.dtype is not torch.bfloat16 or index_k.dtype is not torch.bfloat16:
        raise ValueError(f"DeepSeek V4 Triton indexer requires bfloat16 q/k, got q={index_q.dtype}, k={index_k.dtype}")
    if weights.dtype is not torch.float32:
        raise ValueError(f"DeepSeek V4 Triton indexer requires float32 weights, got {weights.dtype}")
    if topk_indices is not None:
        if topk_indices.ndim != 3 or topk_indices.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"DeepSeek V4 Triton indexer requires topk_indices [B={batch}, S={seq_len}, K], "
                f"got {tuple(topk_indices.shape)}"
            )
        if topk_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"DeepSeek V4 Triton indexer requires int32/int64 topk_indices, got {topk_indices.dtype}")
        if topk_indices.device != index_q.device:
            raise ValueError(
                "DeepSeek V4 Triton indexer requires topk_indices on the q device, got "
                f"q={index_q.device}, indices={topk_indices.device}"
            )
    if (cu_seqlen_ks is None) != (cu_seqlen_ke is None):
        raise ValueError("cu_seqlen_ks and cu_seqlen_ke must be provided together")
    if cu_seqlen_ks is None:
        positions = torch.arange(seq_len, device=index_q.device, dtype=torch.int32)
        cu_seqlen_ks = torch.zeros_like(positions)
        cu_seqlen_ke = (positions + 1) // compress_ratio
    else:
        if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
            raise ValueError(
                "Packed indexer ranges must have shape "
                f"({seq_len},), got {tuple(cu_seqlen_ks.shape)} and {tuple(cu_seqlen_ke.shape)}"
            )
        if cu_seqlen_ks.dtype not in (torch.int32, torch.int64) or cu_seqlen_ke.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "DeepSeek V4 Triton indexer requires int32/int64 packed ranges, got "
                f"{cu_seqlen_ks.dtype} and {cu_seqlen_ke.dtype}"
            )
        if cu_seqlen_ks.device != index_q.device or cu_seqlen_ke.device != index_q.device:
            raise ValueError(
                "DeepSeek V4 Triton indexer requires packed ranges on the q device, got "
                f"q={index_q.device}, ks={cu_seqlen_ks.device}, ke={cu_seqlen_ke.device}"
            )
    return _TritonIndexer.apply(
        index_q.transpose(0, 1).contiguous(),
        index_k.transpose(0, 1).contiguous(),
        weights.transpose(0, 1).contiguous(),
        cu_seqlen_ks.to(device=index_q.device, dtype=torch.int32).contiguous(),
        cu_seqlen_ke.to(device=index_q.device, dtype=torch.int32).contiguous(),
        topk,
        topk_indices,
    )
