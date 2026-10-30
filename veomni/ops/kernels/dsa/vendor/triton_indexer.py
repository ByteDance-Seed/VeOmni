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
and skips pool tiles the query cannot see; ``torch.topk`` selects. Only the
selected scores are differentiable, so the backward recomputes the selected
logits per query. Neither direction materializes ``[B, S, H, P]``.
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
    offs_p = p0 + tl.arange(0, BLOCK_P)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    p_mask = offs_p < P
    h_mask = offs_h < H
    k = tl.load(K_PTR + (b * P + offs_p[:, None]) * D + offs_d[None, :], mask=p_mask[:, None], other=0.0)
    for i in range(BLOCK_S):
        s = s0 + i
        if s < S:
            row = b * S + s
            ks = tl.load(KS_PTR + s)
            ke = tl.load(KE_PTR + s)
            if (p0 < ke) & (p0 + BLOCK_P > ks):
                q = tl.load(
                    Q_PTR + row * H * D + offs_h[:, None] * D + offs_d[None, :], mask=h_mask[:, None], other=0.0
                )
                w = tl.load(W_PTR + row * H + offs_h, mask=h_mask, other=0.0)
                score = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)
                visible = (offs_p >= ks) & (offs_p < ke)
                tl.store(OUT_PTR + row * P + offs_p, tl.where(visible, score, float("-inf")), mask=p_mask)
            else:
                tl.store(OUT_PTR + row * P + offs_p, tl.full((BLOCK_P,), float("-inf"), tl.float32), mask=p_mask)


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
        valid = idx >= 0
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


def _visible_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor):
    """``[B, S, P]`` FP32 scores, ``-inf`` outside each query's ``[ks, ke)``."""
    batch, seq_len, heads, dim = q.shape
    pool = k.shape[1]
    scores = torch.empty((batch, seq_len, pool), dtype=torch.float32, device=q.device)
    _scores_kernel[(batch * triton.cdiv(seq_len, _SCORES_BLOCK_S), triton.cdiv(pool, _SCORES_BLOCK_P))](
        q,
        k,
        w,
        ks,
        ke,
        scores,
        seq_len,
        pool,
        heads,
        D=dim,
        BLOCK_H=max(16, triton.next_power_of_2(heads)),
        BLOCK_S=_SCORES_BLOCK_S,
        BLOCK_P=_SCORES_BLOCK_P,
        num_warps=_SCORES_NUM_WARPS,
        num_stages=1,
    )
    return scores


def _gather_scores(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    valid = (indices >= 0) & (indices < scores.shape[-1])
    safe = indices.clamp(min=0, max=max(scores.shape[-1] - 1, 0)).to(torch.int64)
    return torch.where(valid, torch.gather(scores, -1, safe), float("-inf"))


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

        scores = _visible_scores(q, k, w, ks, ke)
        if topk_indices is None:
            values, indices = scores.topk(min(topk, pool), dim=-1)
            indices = torch.where(torch.isfinite(values), indices, -1).to(torch.int32)
        else:
            indices = topk_indices
        selected = _gather_scores(scores, indices)
        visible_indices, _ = mask_topk_indices_to_query_range(indices.to(torch.int32), ks, ke)
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
    seq_len, _, heads, dim = index_q.shape
    if heads > 64:
        raise ValueError(f"DeepSeek V4 Triton indexer supports at most 64 heads, got {heads}")
    if dim < 16 or dim & (dim - 1):
        raise ValueError(f"DeepSeek V4 Triton indexer requires a power-of-two head dim >= 16, got {dim}")
    if index_q.dtype is not torch.bfloat16 or index_k.dtype is not torch.bfloat16:
        raise ValueError(f"DeepSeek V4 Triton indexer requires bfloat16 q/k, got q={index_q.dtype}, k={index_k.dtype}")
    if weights.dtype is not torch.float32:
        raise ValueError(f"DeepSeek V4 Triton indexer requires float32 weights, got {weights.dtype}")
    if (cu_seqlen_ks is None) != (cu_seqlen_ke is None):
        raise ValueError("cu_seqlen_ks and cu_seqlen_ke must be provided together")
    if cu_seqlen_ks is None:
        positions = torch.arange(seq_len, device=index_q.device, dtype=torch.int32)
        cu_seqlen_ks = torch.zeros_like(positions)
        cu_seqlen_ke = (positions + 1) // compress_ratio
    elif cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            "Packed indexer ranges must have shape "
            f"({seq_len},), got {tuple(cu_seqlen_ks.shape)} and {tuple(cu_seqlen_ke.shape)}"
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
