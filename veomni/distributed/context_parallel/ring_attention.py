# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Ported/adapted from MindSpeed AttentionWithCp (BSD-3-Clause) for Open-VeOmni.
"""Minimal causal Ring context-parallel attention (CP>=2, BNSD, dropout=0)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from .attention_backend import (
    has_npu_fusion_attention,
    npu_attention_backward,
    npu_attention_forward,
    torch_attention_backward,
    torch_attention_forward,
)
from .causal_schedule import (
    as_balanced_halves,
    causal_backward_fetch,
    causal_forward_fetch,
    causal_grad_update,
    causal_out_update,
    flatten_balanced_halves,
)
from .ring_p2p import RingP2P
from .sharding import balanced_cp_restore, balanced_cp_slice


def _attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    num_heads: int,
    softmax_scale: float,
    causal: bool,
    backend: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if backend == "npu":
        return npu_attention_forward(
            query,
            key,
            value,
            num_heads=num_heads,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return torch_attention_forward(query, key, value, softmax_scale=softmax_scale, causal=causal)


def _attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dout: Tensor,
    attn_out: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    *,
    num_heads: int,
    softmax_scale: float,
    causal: bool,
    backend: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if backend == "npu":
        return npu_attention_backward(
            query,
            key,
            value,
            dout,
            attn_out,
            softmax_max,
            softmax_sum,
            num_heads=num_heads,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return torch_attention_backward(
        query,
        key,
        value,
        dout,
        attn_out,
        softmax_max,
        softmax_sum,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _default_backend(requested: Optional[str]) -> str:
    if requested is not None:
        return requested
    return "npu" if has_npu_fusion_attention() else "torch"


def simulate_ring_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    cp_size: int,
    softmax_scale: Optional[float] = None,
    backend: str = "torch",
) -> Tensor:
    """Single-process causal Ring CP simulation with full KV visibility.

    Used for Day-1 schedule/online-softmax correctness against dense attention.
    Inputs are global BNSD tensors; output restores canonical sequence order.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}.")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    num_heads = query.shape[1]
    local_outputs = []
    for cp_rank in range(cp_size):
        local_q = as_balanced_halves(balanced_cp_slice(query, cp_size, cp_rank, dim=2))
        local_k = as_balanced_halves(balanced_cp_slice(key, cp_size, cp_rank, dim=2))
        local_v = as_balanced_halves(balanced_cp_slice(value, cp_size, cp_rank, dim=2))

        # Collect every rank's balanced KV shard so we can emulate the ring.
        kv_shards = [
            (
                as_balanced_halves(balanced_cp_slice(key, cp_size, rank, dim=2)),
                as_balanced_halves(balanced_cp_slice(value, cp_size, rank, dim=2)),
            )
            for rank in range(cp_size)
        ]

        attn_out = softmax_max = softmax_sum = None
        kv_block_id = cp_rank
        for _ in range(cp_size):
            cur_k, cur_v = kv_shards[kv_block_id]
            cur_q, cur_k, cur_v, cur_mask = causal_forward_fetch(
                cp_rank, kv_block_id, local_q, cur_k, cur_v, attn_mask=True
            )
            step_out, step_max, step_sum = _attention_forward(
                cur_q,
                cur_k,
                cur_v,
                num_heads=num_heads,
                softmax_scale=softmax_scale,
                causal=cur_mask is not None,
                backend=backend,
            )
            attn_out, softmax_max, softmax_sum = causal_out_update(
                cp_rank,
                kv_block_id,
                step_out,
                step_max,
                step_sum,
                attn_out,
                softmax_max,
                softmax_sum,
            )
            kv_block_id = (kv_block_id - 1) % cp_size

        local_outputs.append(attn_out)

    return balanced_cp_restore(torch.cat(local_outputs, dim=2), cp_size=cp_size, dim=2)


class AttentionWithCp(torch.autograd.Function):
    """Causal Ring attention with context parallelism (single ring, full KV cache)."""

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        cp_group: dist.ProcessGroup,
        cp_global_ranks: Sequence[int],
        softmax_scale: Optional[float] = None,
        backend: Optional[str] = None,
        overlap_group: Optional[dist.ProcessGroup] = None,
    ) -> Tensor:
        backend = _default_backend(backend)
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        cp_size = dist.get_world_size(cp_group)
        rank = dist.get_rank(cp_group)
        if cp_size < 2:
            raise ValueError("AttentionWithCp requires cp_size >= 2.")

        # Local shard is already balanced CP-sliced by the caller: [B,H,2S,D]
        q = as_balanced_halves(query)
        k = as_balanced_halves(key)
        v = as_balanced_halves(value)

        ring = RingP2P(cp_global_ranks, cp_group, overlap_group)
        cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        next_kv = torch.empty_like(cur_kv)

        attn_out = softmax_max = softmax_sum = None
        kv_block_id = rank
        cached_k = []
        cached_v = []

        for step in range(cp_size):
            if step < cp_size - 1:
                ring.async_send_recv(cur_kv, next_kv)

            cur_k, cur_v = cur_kv[0], cur_kv[1]
            cached_k.append(cur_k.clone())
            cached_v.append(cur_v.clone())

            cur_q, step_k, step_v, cur_mask = causal_forward_fetch(
                rank, kv_block_id, q, cur_k, cur_v, attn_mask=True
            )
            step_out, step_max, step_sum = _attention_forward(
                cur_q,
                step_k,
                step_v,
                num_heads=num_heads,
                softmax_scale=softmax_scale,
                causal=cur_mask is not None,
                backend=backend,
            )
            attn_out, softmax_max, softmax_sum = causal_out_update(
                rank,
                kv_block_id,
                step_out,
                step_max,
                step_sum,
                attn_out,
                softmax_max,
                softmax_sum,
            )

            if ring.wait():
                cur_kv, next_kv = next_kv, cur_kv
            kv_block_id = (kv_block_id - 1) % cp_size

        # Save KV in reverse visitation order for backward replay:
        # forward visits rank, rank-1, ..., so cache[0]=local, cache[-1]=last received.
        # Backward visits in reverse: start from last received KV.
        k_stack = torch.stack(cached_k[::-1], dim=0)
        v_stack = torch.stack(cached_v[::-1], dim=0)
        ctx.save_for_backward(q, k_stack, v_stack, attn_out, softmax_max, softmax_sum)
        ctx.num_heads = num_heads
        ctx.softmax_scale = float(softmax_scale)
        ctx.backend = backend
        ctx.cp_size = cp_size
        ctx.rank = rank
        ctx.cp_global_ranks = list(cp_global_ranks)
        ctx.cp_group = cp_group
        ctx.overlap_group = overlap_group
        # Final kv_block_id after forward loop points at the next (unused) block;
        # last computed block was (rank - (cp_size-1)) % cp_size.
        ctx.final_kv_block_id = (rank - (cp_size - 1)) % cp_size
        return attn_out

    @staticmethod
    def backward(ctx, dout: Tensor):
        q, k_stack, v_stack, attn_out, softmax_max, softmax_sum = ctx.saved_tensors
        cp_size = ctx.cp_size
        rank = ctx.rank
        backend = ctx.backend
        softmax_scale = ctx.softmax_scale
        num_heads = ctx.num_heads

        dout_h = as_balanced_halves(dout)
        attn_out_h = as_balanced_halves(attn_out)
        softmax_max_h = as_balanced_halves(softmax_max)
        softmax_sum_h = as_balanced_halves(softmax_sum)

        # Full-cache replay: KV comes from forward saves; only dKV travels the ring.
        dkv_ring = RingP2P(ctx.cp_global_ranks, ctx.cp_group, ctx.overlap_group, is_backward=True)
        cur_dkv = torch.zeros((2, *k_stack[0].shape), dtype=k_stack[0].dtype, device=k_stack[0].device)
        next_dkv = torch.zeros_like(cur_dkv)

        dq = torch.zeros_like(q)
        kv_block_id = ctx.final_kv_block_id

        for step in range(cp_size):
            cur_k = k_stack[step]
            cur_v = v_stack[step]
            step_q, step_k, step_v, step_out, step_dout, step_max, step_sum, step_mask = causal_backward_fetch(
                rank,
                kv_block_id,
                q,
                cur_k,
                cur_v,
                attn_out_h,
                dout_h,
                softmax_max_h,
                softmax_sum_h,
                attn_mask=True,
            )
            dq_step, dk_step, dv_step = _attention_backward(
                step_q,
                step_k,
                step_v,
                step_dout,
                step_out,
                step_max,
                step_sum,
                num_heads=num_heads,
                softmax_scale=softmax_scale,
                causal=step_mask is not None,
                backend=backend,
            )

            if step > 0:
                dkv_ring.wait()
                cur_dkv, next_dkv = next_dkv, cur_dkv

            dk, dv = cur_dkv[0], cur_dkv[1]
            causal_grad_update(rank, kv_block_id, dq_step, dk_step, dv_step, dq, dk, dv)

            if step < cp_size - 1:
                dkv_ring.async_send_recv(cur_dkv, next_dkv)

            kv_block_id = (kv_block_id + 1) % cp_size

        if dkv_ring.wait():
            cur_dkv, next_dkv = next_dkv, cur_dkv

        dk, dv = cur_dkv[0], cur_dkv[1]
        return (
            flatten_balanced_halves(dq),
            flatten_balanced_halves(dk),
            flatten_balanced_halves(dv),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def ringattn_context_parallel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    softmax_scale: Optional[float] = None,
    backend: Optional[str] = None,
    overlap_group: Optional[dist.ProcessGroup] = None,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """Public entry for causal Ring CP attention over already-sliced local Q/K/V.

    When ``cu_seqlens`` is provided, runs per-sample Ring (no cross-sample attention).
    Local sample lengths must already match the packed CP partition.
    """
    if cu_seqlens is None:
        return AttentionWithCp.apply(
            query,
            key,
            value,
            num_heads,
            cp_group,
            list(cp_global_ranks),
            softmax_scale,
            backend,
            overlap_group,
        )

    cu = cu_seqlens.detach().to(device="cpu").tolist()
    if query.size(0) != 1:
        raise ValueError(f"Packed Ring CP currently supports batch=1, got {query.shape}.")
    outs = []
    for start, end in zip(cu[:-1], cu[1:]):
        start, end = int(start), int(end)
        if end <= start:
            continue
        outs.append(
            AttentionWithCp.apply(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                num_heads,
                cp_group,
                list(cp_global_ranks),
                softmax_scale,
                backend,
                overlap_group,
            )
        )
    if not outs:
        return query.new_empty(query.shape)
    return torch.cat(outs, dim=2)


def simulate_packed_ring_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    *,
    cp_size: int,
    softmax_scale: Optional[float] = None,
    backend: str = "torch",
) -> Tensor:
    """Per-sample Ring CP simulation (no cross-sample leakage)."""
    from .attention_backend import torch_packed_causal_attention  # noqa: F401 — kept for API symmetry

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    cu = cu_seqlens.detach().cpu().tolist()
    outs = []
    for start, end in zip(cu[:-1], cu[1:]):
        start, end = int(start), int(end)
        outs.append(
            simulate_ring_causal_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                cp_size=cp_size,
                softmax_scale=softmax_scale,
                backend=backend,
            )
        )
    return torch.cat(outs, dim=2)


def dense_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """Reference dense causal attention (BNSD, GQA-aware)."""
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    out, _, _ = torch_attention_forward(query, key, value, softmax_scale=softmax_scale, causal=True)
    return out
