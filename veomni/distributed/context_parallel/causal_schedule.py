# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Ported/adapted from MindSpeed ring_context_parallel causal helpers (BSD-3-Clause).
"""Causal zigzag block schedule for Ring context parallel (BNSD layout)."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from .softmax_update import merge_attention_blocks


def as_balanced_halves(tensor: Tensor) -> Tensor:
    """View local CP shard [B, H, 2S, ...] as [B, H, 2, S, ...]."""
    if tensor.size(2) % 2 != 0:
        raise ValueError(f"Sequence dim must be even for balanced CP, got {tensor.size(2)}.")
    return tensor.view(tensor.size(0), tensor.size(1), 2, tensor.size(2) // 2, *tensor.shape[3:])


def flatten_balanced_halves(tensor: Tensor) -> Tensor:
    """Flatten [B, H, 2, S, ...] back to [B, H, 2S, ...]."""
    return tensor.reshape(tensor.size(0), tensor.size(1), -1, *tensor.shape[4:])


def causal_forward_fetch(
    q_block_id: int,
    kv_block_id: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Select Q/K/V slices for one causal ring step.

    Inputs are balanced-halved BNSD: [B, H, 2, S, D].
    """
    if q_block_id == kv_block_id:
        return (
            flatten_balanced_halves(query),
            flatten_balanced_halves(key),
            flatten_balanced_halves(value),
            attn_mask,
        )
    if kv_block_id <= q_block_id:
        return (
            flatten_balanced_halves(query),
            key[:, :, 0],
            value[:, :, 0],
            None,
        )
    return (
        query[:, :, 1],
        flatten_balanced_halves(key),
        flatten_balanced_halves(value),
        None,
    )


def causal_backward_fetch(
    q_block_id: int,
    kv_block_id: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_out: Tensor,
    dout: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    attn_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Select backward operands for one causal ring step (balanced-halved BNSD)."""
    if q_block_id >= kv_block_id:
        cur_softmax_max = flatten_balanced_halves(softmax_max)
        cur_softmax_sum = flatten_balanced_halves(softmax_sum)
        cur_q = flatten_balanced_halves(query)
        cur_attn_out = flatten_balanced_halves(attn_out)
        cur_dout = flatten_balanced_halves(dout)
        if q_block_id == kv_block_id:
            return (
                cur_q,
                flatten_balanced_halves(key),
                flatten_balanced_halves(value),
                cur_attn_out,
                cur_dout,
                cur_softmax_max,
                cur_softmax_sum,
                attn_mask,
            )
        return (
            cur_q,
            key[:, :, 0],
            value[:, :, 0],
            cur_attn_out,
            cur_dout,
            cur_softmax_max,
            cur_softmax_sum,
            None,
        )

    return (
        query[:, :, 1],
        flatten_balanced_halves(key),
        flatten_balanced_halves(value),
        attn_out[:, :, 1],
        dout[:, :, 1],
        softmax_max[:, :, 1],
        softmax_sum[:, :, 1],
        None,
    )


def causal_out_update(
    q_block_id: int,
    kv_block_id: int,
    cur_attn_out: Tensor,
    cur_softmax_max: Tensor,
    cur_softmax_sum: Tensor,
    attn_out: Optional[Tensor],
    softmax_max: Optional[Tensor],
    softmax_sum: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Merge one causal ring step into the running online-softmax state."""
    if q_block_id == kv_block_id or attn_out is None:
        return cur_attn_out, cur_softmax_max, cur_softmax_sum

    if kv_block_id <= q_block_id:
        return merge_attention_blocks(
            attn_out,
            softmax_max,
            softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
        )

    # Future relative block: only the late query half observes this KV.
    out_view = as_balanced_halves(attn_out)
    max_view = as_balanced_halves(softmax_max)
    sum_view = as_balanced_halves(softmax_sum)
    updated_out, updated_max, updated_sum = merge_attention_blocks(
        out_view[:, :, 1],
        max_view[:, :, 1],
        sum_view[:, :, 1],
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
    )
    out_view = out_view.clone()
    max_view = max_view.clone()
    sum_view = sum_view.clone()
    out_view[:, :, 1].copy_(updated_out)
    max_view[:, :, 1].copy_(updated_max)
    sum_view[:, :, 1].copy_(updated_sum)
    return (
        flatten_balanced_halves(out_view),
        flatten_balanced_halves(max_view),
        flatten_balanced_halves(sum_view),
    )


def causal_grad_update(
    q_block_id: int,
    kv_block_id: int,
    cur_dq: Tensor,
    cur_dk: Tensor,
    cur_dv: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
) -> None:
    """Accumulate step gradients into balanced-halved dQ/dK/dV buffers."""
    if q_block_id == kv_block_id:
        dq.add_(as_balanced_halves(cur_dq))
        dk.add_(as_balanced_halves(cur_dk))
        dv.add_(as_balanced_halves(cur_dv))
        return

    if q_block_id > kv_block_id:
        dq.add_(as_balanced_halves(cur_dq))
        dk[:, :, 0].add_(cur_dk)
        dv[:, :, 0].add_(cur_dv)
        return

    dq[:, :, 1].add_(cur_dq)
    dk.add_(as_balanced_halves(cur_dk))
    dv.add_(as_balanced_halves(cur_dv))
