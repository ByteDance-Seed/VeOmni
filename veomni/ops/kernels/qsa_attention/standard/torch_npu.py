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

"""standard QSA bounded-mask Ascend NPU adapter (``torch_npu``)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .....utils import logging
from . import eager


logger = logging.get_logger(__name__)


def _validate_qsa_inputs(query: Tensor, key: Tensor, value: Tensor, selected_indices: Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"query/key/value must be rank-4 tensors; got query={query.shape}, key={key.shape}, value={value.shape}"
        )
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes must match; got key={key.shape}, value={value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key batch and head dimensions must match; got query={query.shape}, key={key.shape}")
    if selected_indices.ndim != 3 or selected_indices.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError(
            "query batch and sequence dimensions must match selected_indices; "
            f"got query={query.shape}, selected_indices={selected_indices.shape}"
        )
    if not (query.device == key.device == value.device == selected_indices.device):
        raise ValueError("query/key/value/selected_indices must be on the same device")
    if any(size == 0 for size in query.shape) or key.shape[1] == 0:
        raise ValueError("query dimensions and key/value heads must be positive")
    if selected_indices.shape[-1] == 0:
        raise ValueError("selected_indices must contain at least one slot (use -1 for empty selections)")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(f"query heads ({query.shape[1]}) must be divisible by kv heads ({key.shape[1]})")
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_indices must use int32 or int64, got {selected_indices.dtype}")
    if key.shape[2] == 0:
        raise ValueError("key/value must contain at least one token")
    if torch.any((selected_indices < -1) | (selected_indices >= key.shape[2])):
        raise ValueError(
            f"selected_indices must contain -1 padding or valid global KV token indices in [0, {key.shape[2]})"
        )


def _blocked_mask(indices: Tensor, kv_length: int) -> Tensor:
    # An extra column absorbs -1 padding without overwriting valid duplicates.
    blocked = torch.ones((*indices.shape[:-1], kv_length + 1), dtype=torch.int8, device=indices.device)
    slots = torch.where(indices >= 0, indices, kv_length).long()
    blocked.scatter_(-1, slots, 0)
    return blocked[..., :kv_length].unsqueeze(1).bool().contiguous()


def _kv_ranges(indices: Tensor, kv_length: int, chunk_size: int) -> list[tuple[int, int]]:
    # One host synchronization per attention call, not per query block.
    bounds = []
    for start in range(0, indices.shape[1], chunk_size):
        block = indices[:, start : start + chunk_size].to(torch.int32)
        lo = block.masked_fill(block < 0, kv_length).amin()
        hi = block.amax() + 1
        bounds.append(torch.stack((lo, hi)))
    ranges = torch.stack(bounds).cpu().tolist()
    # Alignment improves native kernel tiling. Never exclude a selected token.
    return [
        (max(0, lo // 128 * 128), min(kv_length, (hi + 127) // 128 * 128)) if hi > 0 else (0, 1) for lo, hi in ranges
    ]


def _forward(q: Tensor, k: Tensor, v: Tensor, blocked: Tensor, scale: float):
    import torch_npu

    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        "BNSD",
        atten_mask=blocked,
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
        inner_precise=2,
    )


def _backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grad: Tensor,
    blocked: Tensor,
    output: Tensor,
    maximum: Tensor,
    denominator: Tensor,
    scale: float,
):
    import torch_npu

    return torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        grad,
        q.shape[1],
        "BNSD",
        atten_mask=blocked,
        softmax_max=maximum,
        softmax_sum=denominator,
        attention_in=output,
        scale_value=scale,
        keep_prob=1.0,
        sparse_mode=0,
        inner_precise=2,
    )[:3]


class _CompactFusedAttention(torch.autograd.Function):
    """Chunked Ascend fused attention over per-block KV intervals and int8 scatter masks."""

    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, indices: Tensor, scale: float, chunk_size: int):
        """Run chunked BNSD fused attention over pruned KV intervals and save softmax statistics."""
        key, value = key.contiguous(), value.contiguous()
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        statistics = []
        ranges = _kv_ranges(indices, key.shape[2], chunk_size)
        for block, start in enumerate(range(0, query.shape[2], chunk_size)):
            section = slice(start, start + chunk_size)
            q = query[:, :, section].contiguous()
            lo, hi = ranges[block]
            local_indices = indices[:, section] - lo
            blocked = _blocked_mask(local_indices, hi - lo)
            result, maximum, denominator, *_ = _forward(
                q, key[:, :, lo:hi].contiguous(), value[:, :, lo:hi].contiguous(), blocked, scale
            )
            valid_rows = (indices[:, section] >= 0).any(-1)[:, None, :, None]
            result.masked_fill_(~valid_rows, 0)
            output[:, :, section].copy_(result)
            statistics.extend((maximum, denominator))
        ctx.save_for_backward(query, key, value, indices, output, *statistics)
        ctx.ranges = ranges
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        """Rebuild per-chunk masks and accumulate KV gradients in FP32 without forward replay."""
        query, key, value, indices, output, *statistics = ctx.saved_tensors
        grad_query = torch.empty_like(query, memory_format=torch.contiguous_format)
        # Accumulate disjoint-query contributions before a single BF16 writeback.
        grad_key = torch.zeros_like(key, dtype=torch.float32)
        grad_value = torch.zeros_like(value, dtype=torch.float32)
        for block, start in enumerate(range(0, query.shape[2], ctx.chunk_size)):
            section = slice(start, start + ctx.chunk_size)
            lo, hi = ctx.ranges[block]
            blocked = _blocked_mask(indices[:, section] - lo, hi - lo)
            valid_rows = (indices[:, section] >= 0).any(-1)[:, None, :, None]
            dq, dk, dv = _backward(
                query[:, :, section].contiguous(),
                key[:, :, lo:hi].contiguous(),
                value[:, :, lo:hi].contiguous(),
                grad_output[:, :, section].masked_fill(~valid_rows, 0).contiguous(),
                blocked,
                output[:, :, section].contiguous(),
                statistics[2 * block],
                statistics[2 * block + 1],
                ctx.scale,
            )
            # CANN may return subnormals for fully masked rows. Their output
            # is constant zero, so its derivative must also be exactly zero.
            dq.masked_fill_(~valid_rows, 0)
            empty_batch = ~valid_rows.any(-2, keepdim=True)
            dk.masked_fill_(empty_batch, 0)
            dv.masked_fill_(empty_batch, 0)
            grad_query[:, :, section].copy_(dq)
            grad_key[:, :, lo:hi].add_(dk)
            grad_value[:, :, lo:hi].add_(dv)
        return grad_query, grad_key.to(key.dtype), grad_value.to(value.dtype), None, None, None


def qsa_attn_npu_fused(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_indices: Tensor,
    sm_scale: float | None = None,
    query_chunk_size: int = 2048,
) -> Tensor:
    """Return ``[B, S, H, D]`` QSA output on Ascend NPU, preserving global selection semantics.

    No dense masks are retained across query blocks or forward/backward. Native
    GQA keeps KV heads unexpanded. This is dense fused attention over bounded
    per-chunk KV intervals, not a sparse-FLOP kernel, and does not change
    CP/Ulysses communication or checkpoint keys.
    """
    if query.device.type != "npu":
        raise RuntimeError("QSA torch_npu requires Ascend NPU tensors")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("QSA torch_npu requires BF16 query/key/value")
    _validate_qsa_inputs(query, key, value, selected_indices)
    if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
        raise ValueError("QSA query_chunk_size must be a positive integer")
    scale = query.shape[-1] ** -0.5 if sm_scale is None else sm_scale
    result = _CompactFusedAttention.apply(query, key, value, selected_indices, scale, query_chunk_size)
    return result.transpose(1, 2).contiguous()


def wrapper(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_indices: Tensor,
    *,
    scaling: float,
    dropout: float = 0.0,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, None]:
    """Run bounded-mask Ascend fused attention over the compact QSA selection.

    The contract matches ``eager`` and ``tilelang``: *query*, *key*, and *value*
    arrive in ``[B, H, S, D]`` layout, *selected_indices* is ``[B, S, K]`` with
    ``-1`` padding, and the returned output is ``[B, S, H, D]`` with ``None``
    for attention weights.

    The selection already encodes causality and packed-sample boundaries, so an
    additive or boolean *attention_mask* raises. Nonzero *dropout* falls back
    to the dense eager reference with a one-time warning.
    """
    if attention_mask is not None:
        raise ValueError(
            f"torch_npu QSA attention requires no attention_mask; got attention_mask={type(attention_mask).__name__}."
        )
    if dropout:
        logger.warning_once(
            f"torch_npu QSA attention has no dropout path; attention_dropout={dropout} falls back to the "
            "dense eager QSA reference, which materializes the [B, H, S, S] score tensor."
        )
        return eager.wrapper(query, key, value, selected_indices, scaling=scaling, dropout=dropout)

    return qsa_attn_npu_fused(query, key, value, selected_indices, sm_scale=scaling), None
