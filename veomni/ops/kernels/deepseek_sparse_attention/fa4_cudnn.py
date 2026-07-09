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

"""FlashAttention-4 forward paired with cuDNN FE DSA backward."""

from __future__ import annotations

from typing import Any

import torch
from flash_attn.cute import flash_attn_func

from veomni.ops.kernels.deepseek_sparse_attention.common import (
    check_sparse_mla_forward_compatible,
    pack_sparse_mla_tensors_for_backward,
    sparse_attention_backward,
)


def fa4_sparse_forward(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope: torch.Tensor,
    gather_kv_indices: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    min_seqlen_k: int | None = None,
    learnable_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Run FA4 sparse MLA forward and return both output and LSE."""
    compatible, reason = check_sparse_mla_forward_compatible(
        q_pe,
        k_pe,
        kv_cache,
        q_nope,
        gather_kv_indices,
        learnable_sink,
        topk_length,
    )
    if not compatible:
        raise ValueError(reason)
    if gather_kv_indices is None:
        raise ValueError("FA4 sparse MLA forward requires gather_kv_indices")
    if causal:
        raise ValueError("FA4 sparse MLA forward requires causal=False")
    if min_seqlen_k is not None:
        raise ValueError("FA4 sparse MLA forward does not use min_seqlen_k")
    if kwargs:
        raise ValueError(f"Unsupported FA4 sparse MLA forward kwargs: {sorted(kwargs)}")

    sm_scale = (q_pe.shape[-1] + q_nope.shape[-1]) ** (-0.5) if softmax_scale is None else softmax_scale
    indices = gather_kv_indices.to(torch.int32)
    if topk_length is not None:
        positions = torch.arange(indices.shape[-1], device=indices.device, dtype=topk_length.dtype)
        valid = positions.view(1, 1, -1) < topk_length.unsqueeze(-1)
        indices = torch.where(valid, indices, torch.full_like(indices, -1))
    result = flash_attn_func(
        q_pe,
        k_pe,
        kv_cache,
        qv=q_nope,
        softmax_scale=sm_scale,
        causal=False,
        learnable_sink=learnable_sink,
        gather_kv_indices=indices,
        return_lse=True,
    )
    out, lse = result[:2]
    return {"out": out, "lse": lse}


class _FA4SparseAttentionWithCuDNNBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        q_nope_absorbed: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_sink: torch.Tensor | None,
        topk_length: torch.Tensor | None,
        softmax_scale: float | None,
    ) -> torch.Tensor:
        topk_indices = topk_indices if topk_indices.dtype == torch.int32 else topk_indices.to(torch.int32)
        forward_result = fa4_sparse_forward(
            q_pe,
            k_pe,
            kv_cache,
            q_nope_absorbed,
            topk_indices,
            softmax_scale=softmax_scale,
            learnable_sink=attn_sink,
            topk_length=topk_length,
        )
        saved_tensors = [
            q_pe,
            k_pe,
            kv_cache,
            q_nope_absorbed,
            topk_indices,
            forward_result["out"],
            forward_result["lse"],
        ]
        if attn_sink is not None:
            saved_tensors.append(attn_sink)
        if topk_length is not None:
            saved_tensors.append(topk_length)
        ctx.save_for_backward(*saved_tensors)
        ctx.has_attn_sink = attn_sink is not None
        ctx.has_topk_length = topk_length is not None
        ctx.softmax_scale = softmax_scale
        return forward_result["out"]

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        saved = ctx.saved_tensors
        q_pe, k_pe, kv_cache, q_nope_absorbed, topk_indices, out, lse = saved[:7]
        offset = 7
        if ctx.has_attn_sink:
            attn_sink = saved[offset]
            offset += 1
        else:
            attn_sink = torch.full((q_pe.shape[2],), float("-inf"), device=q_pe.device, dtype=torch.float32)
        topk_length = saved[offset] if ctx.has_topk_length else None
        packed = pack_sparse_mla_tensors_for_backward(q_pe, k_pe, kv_cache, q_nope_absorbed)

        backward_result = sparse_attention_backward(
            packed["q"],
            packed["kv"],
            out,
            dout,
            lse,
            attn_sink,
            topk_indices,
            softmax_scale=ctx.softmax_scale,
            topk_length=topk_length,
        )
        dq_nope_absorbed, dq_pe = torch.split(
            backward_result["dq"], [q_nope_absorbed.shape[-1], q_pe.shape[-1]], dim=-1
        )
        dkv_cache, dk_pe = torch.split(backward_result["dkv"], [kv_cache.shape[-1], k_pe.shape[-1]], dim=-1)

        return (
            dq_pe,
            dk_pe.unsqueeze(2),
            dkv_cache.unsqueeze(2),
            dq_nope_absorbed,
            None,
            backward_result["d_sink"] if ctx.has_attn_sink else None,
            None,
            None,
        )


def fa4_sparse_attention_with_cudnn_backward(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope_absorbed: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """FA4 sparse MLA forward paired with cuDNN FE DSA backward."""
    return _FA4SparseAttentionWithCuDNNBackward.apply(
        q_pe,
        k_pe,
        kv_cache,
        q_nope_absorbed,
        topk_indices,
        attn_sink,
        topk_length,
        softmax_scale,
    )


__all__ = [
    "fa4_sparse_forward",
    "fa4_sparse_attention_with_cudnn_backward",
]
