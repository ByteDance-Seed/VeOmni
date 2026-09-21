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

"""Thin VeOmni entrypoints for cuDNN Frontend DeepSeek Sparse Attention.

The implementation lives in NVIDIA's ``nvidia-cudnn-frontend`` package under
``cudnn.DSA`` and uses FlashMLA sparse prefill for the forward path.
"""

from __future__ import annotations

from typing import Any

import torch
from cudnn import DSA
from flash_mla import flash_mla_sparse_fwd


def _local_topk_to_global(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    if topk_indices.dim() != 3:
        raise ValueError(f"topk_indices must be [B, S_q, topk], got {tuple(topk_indices.shape)}")
    batch_offsets = torch.arange(topk_indices.shape[0], device=topk_indices.device, dtype=torch.int32).view(-1, 1, 1)
    batch_offsets = batch_offsets * int(seqlen_k)
    topk_i32 = topk_indices.to(torch.int32)
    return torch.where(topk_i32 >= 0, topk_i32 + batch_offsets, topk_i32)


def check_sparse_attention_backward_compatible(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_length: torch.Tensor | None = None,
) -> tuple[bool, str]:
    """Validate the tensor contract required by cuDNN FE DSA backward.

    This kernel is not a generic sparse-attention backward. It expects the
    DeepSeek/FlashMLA layout: multi-head queries, a single shared K=V tensor
    (MQA), FlashMLA-style KV-only LSE, and top-k indices over that shared KV.
    """
    if q.dim() != 4:
        return False, f"q must be [B, S_q, H, D], got {tuple(q.shape)}"
    if kv.dim() != 3:
        return False, f"kv must be unified K=V [B, S_kv, D], got {tuple(kv.shape)}"
    if q.shape[0] != kv.shape[0]:
        return False, "q and kv batch sizes must match"
    if kv.shape[-1] != q.shape[-1]:
        return False, f"kv dim must match q dim for unified K=V DSA, got {kv.shape[-1]} and {q.shape[-1]}"
    if out.shape[:3] != q.shape[:3] or dout.shape[:3] != q.shape[:3]:
        return False, "out and dout must have leading shape [B, S_q, H]"
    expected_value_dim = 512 if q.shape[-1] == 576 else q.shape[-1]
    if out.shape[-1] != expected_value_dim or dout.shape[-1] != expected_value_dim:
        return False, f"out/dout value dim must be {expected_value_dim}, got {out.shape[-1]} and {dout.shape[-1]}"
    if lse.shape != q.shape[:3]:
        return False, f"lse must be [B, S_q, H], got {tuple(lse.shape)}"
    if attn_sink.shape != (q.shape[2],):
        return False, f"attn_sink must be [H], got {tuple(attn_sink.shape)}"
    if topk_indices.dim() != 3 or topk_indices.shape[:2] != q.shape[:2]:
        return False, f"topk_indices must be [B, S_q, topk], got {tuple(topk_indices.shape)}"
    if topk_length is not None and topk_length.shape != topk_indices.shape[:2]:
        return False, f"topk_length must be [B, S_q], got {tuple(topk_length.shape)}"
    return True, ""


def check_flash_mla_sparse_forward_compatible(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope: torch.Tensor,
    gather_kv_indices: torch.Tensor | None,
    learnable_sink: torch.Tensor | None = None,
) -> tuple[bool, str]:
    """Validate the FlashMLA sparse prefill contract used by DeepSeek DSA."""
    if q_pe.dim() != 4:
        return False, f"q_pe must be [B, S_q, H, D_pe], got {tuple(q_pe.shape)}"
    if q_nope.shape[:-1] != q_pe.shape[:-1]:
        return False, f"q_nope leading shape must match q_pe, got {tuple(q_nope.shape)} and {tuple(q_pe.shape)}"
    if k_pe.dim() != 4 or kv_cache.dim() != 4:
        return (
            False,
            f"k_pe and kv_cache must be [B, S_kv, H_kv, D], got {tuple(k_pe.shape)} and {tuple(kv_cache.shape)}",
        )
    if k_pe.shape[:3] != kv_cache.shape[:3]:
        return False, "k_pe and kv_cache leading shapes must match"
    if q_pe.shape[0] != k_pe.shape[0]:
        return False, "q_pe and k_pe batch sizes must match"
    if q_pe.shape[-1] != k_pe.shape[-1]:
        return False, f"q_pe/k_pe head dims must match, got {q_pe.shape[-1]} and {k_pe.shape[-1]}"
    if q_nope.shape[-1] != kv_cache.shape[-1]:
        return (
            False,
            f"q_nope and kv_cache value dims must match, got {q_nope.shape[-1]} and {kv_cache.shape[-1]}",
        )
    if kv_cache.shape[-1] != 512:
        return False, f"FlashMLA sparse prefill currently requires value dim 512, got {kv_cache.shape[-1]}"
    packed_dim = q_nope.shape[-1] + q_pe.shape[-1]
    if packed_dim != 576:
        return False, f"FlashMLA sparse prefill currently requires packed q/k dim 576, got {packed_dim}"
    if k_pe.shape[2] != 1:
        return False, f"FlashMLA sparse prefill wrapper requires MQA K/V with H_kv=1, got {k_pe.shape[2]}"
    if gather_kv_indices is not None:
        if gather_kv_indices.shape != (*q_pe.shape[:2], gather_kv_indices.shape[-1]):
            return False, f"gather_kv_indices must be [B, S_q, topk], got {tuple(gather_kv_indices.shape)}"
        if gather_kv_indices.shape[-1] % 128 != 0:
            return (
                False,
                f"FlashMLA sparse prefill requires topk to be a multiple of 128, got {gather_kv_indices.shape[-1]}",
            )
    if learnable_sink is not None:
        return False, "this FlashMLA/cuDNN composite path does not support learnable_sink"
    return True, ""


def flash_mla_sparse_forward(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope: torch.Tensor,
    gather_kv_indices: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    min_seqlen_k: int | None = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Run FlashMLA sparse prefill and return both output and LSE.

    The returned LSE uses FlashMLA's sparse-prefill layout ``[B, S_q, H]``, which can be
    flattened to cuDNN FE DSA backward's ``[total_S_q, H]`` contract. FlashMLA's
    learnable sink support is intentionally not wired through this composite
    path because the backward side would also need matching sink gradients.
    """
    compatible, reason = check_flash_mla_sparse_forward_compatible(
        q_pe,
        k_pe,
        kv_cache,
        q_nope,
        gather_kv_indices,
        kwargs.get("learnable_sink"),
    )
    if not compatible:
        raise ValueError(reason)
    if gather_kv_indices is None:
        raise ValueError("FlashMLA sparse prefill requires gather_kv_indices")

    if causal:
        raise ValueError("FlashMLA sparse prefill requires causal=False")
    if min_seqlen_k is not None:
        raise ValueError("FlashMLA sparse prefill does not use min_seqlen_k")
    if kwargs:
        raise ValueError(f"Unsupported FlashMLA sparse prefill kwargs: {sorted(kwargs)}")

    packed = pack_flash_mla_tensors_for_sparse_backward(q_pe, k_pe, kv_cache, q_nope)
    batch_size, seqlen_q, num_heads = q_pe.shape[:3]
    seqlen_k = k_pe.shape[1]
    q_flat = packed["q"].reshape(batch_size * seqlen_q, num_heads, packed["q"].shape[-1]).contiguous()
    kv_flat = packed["kv"].reshape(batch_size * seqlen_k, 1, packed["kv"].shape[-1]).contiguous()
    indices = _local_topk_to_global(gather_kv_indices, seqlen_k).reshape(batch_size * seqlen_q, 1, -1).contiguous()
    sm_scale = q_flat.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale

    out, _, lse = flash_mla_sparse_fwd(
        q_flat,
        kv_flat,
        indices,
        sm_scale,
        d_v=kv_cache.shape[-1],
    )
    return {
        "out": out.reshape(batch_size, seqlen_q, num_heads, kv_cache.shape[-1]),
        "lse": lse.reshape(batch_size, seqlen_q, num_heads),
    }


def pack_flash_mla_tensors_for_sparse_backward(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pack FlashMLA tensors into cuDNN FE DSA backward's unified layout."""
    compatible, reason = check_flash_mla_sparse_forward_compatible(
        q_pe, k_pe, kv_cache, q_nope, gather_kv_indices=None
    )
    if not compatible:
        raise ValueError(reason)
    return {
        "q": torch.cat((q_nope, q_pe), dim=-1),
        "kv": torch.cat((kv_cache.squeeze(2), k_pe.squeeze(2)), dim=-1),
    }


class _FlashMLASparseAttentionWithCuDNNBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        q_nope_absorbed: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float | None,
    ) -> torch.Tensor:
        forward_result = flash_mla_sparse_forward(
            q_pe,
            k_pe,
            kv_cache,
            q_nope_absorbed,
            topk_indices.to(torch.int32),
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(
            q_pe, k_pe, kv_cache, q_nope_absorbed, topk_indices, forward_result["out"], forward_result["lse"]
        )
        ctx.softmax_scale = softmax_scale
        return forward_result["out"]

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q_pe, k_pe, kv_cache, q_nope_absorbed, topk_indices, out, lse = ctx.saved_tensors
        packed = pack_flash_mla_tensors_for_sparse_backward(q_pe, k_pe, kv_cache, q_nope_absorbed)
        attn_sink = torch.full((q_pe.shape[2],), float("-inf"), device=q_pe.device, dtype=torch.float32)

        backward_result = sparse_attention_backward(
            packed["q"],
            packed["kv"],
            out,
            dout,
            lse,
            attn_sink,
            topk_indices,
            softmax_scale=ctx.softmax_scale,
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
            None,
        )


def flash_mla_sparse_attention_with_cudnn_backward(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_nope_absorbed: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """FlashMLA sparse prefill forward paired with cuDNN FE DSA backward."""
    return _FlashMLASparseAttentionWithCuDNNBackward.apply(
        q_pe,
        k_pe,
        kv_cache,
        q_nope_absorbed,
        topk_indices,
        softmax_scale,
    )


def sparse_attention_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    topk_length: torch.Tensor | None = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Run cuDNN FE DSA backward from batched VeOmni-style tensors.

    cuDNN's DSA backward consumes flattened query rows and global top-k ids,
    while model code naturally carries ``[B, S, ...]`` tensors and local top-k
    indices. This helper is the narrow bridge between those contracts; it still
    expects the caller to provide FlashMLA-compatible forward outputs ``out``
    and ``lse`` and a unified MQA K=V tensor.
    """
    compatible, reason = check_sparse_attention_backward_compatible(
        q, kv, out, dout, lse, attn_sink, topk_indices, topk_length
    )
    if not compatible:
        raise ValueError(reason)

    batch_size, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k = kv.shape[1]
    q_flat = q.reshape(batch_size * seqlen_q, num_heads, head_dim).contiguous()
    kv_flat = kv.reshape(batch_size * seqlen_k, kv.shape[-1]).contiguous()
    out_flat = out.reshape(batch_size * seqlen_q, num_heads, out.shape[-1]).contiguous()
    dout_flat = dout.reshape(batch_size * seqlen_q, num_heads, dout.shape[-1]).contiguous()
    lse_flat = lse.reshape(batch_size * seqlen_q, num_heads).contiguous()
    topk_flat = (
        _local_topk_to_global(topk_indices, seqlen_k)
        .reshape(batch_size * seqlen_q, topk_indices.shape[-1])
        .contiguous()
    )
    topk_length_flat = (
        None if topk_length is None else topk_length.reshape(batch_size * seqlen_q).to(torch.int32).contiguous()
    )

    result = DSA.sparse_attention_backward_wrapper(
        q_flat,
        kv_flat,
        out_flat,
        dout_flat,
        lse_flat,
        attn_sink,
        topk_flat,
        softmax_scale=softmax_scale,
        topk_length=topk_length_flat,
        **kwargs,
    )
    return {
        "dq": result["dq"].reshape_as(q),
        "dkv": result["dkv"].reshape_as(kv),
        "d_sink": result["d_sink"],
    }


def indexer_select_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    top_k: int,
    *,
    ratio: int = 1,
    qhead_per_kv_head: int | None = None,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """Compute DSA indexer scores with cuDNN FE and select local top-k indices."""
    if k.dim() == 3:
        k = k.unsqueeze(2)

    scores = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=sm_scale,
    )["scores"]
    top_k = min(int(top_k), int(scores.shape[-1]))
    return scores.topk(top_k, dim=-1).indices.to(torch.long)


__all__ = [
    "indexer_select_topk",
    "check_flash_mla_sparse_forward_compatible",
    "flash_mla_sparse_forward",
    "pack_flash_mla_tensors_for_sparse_backward",
    "flash_mla_sparse_attention_with_cudnn_backward",
    "check_sparse_attention_backward_compatible",
    "sparse_attention_backward",
]
