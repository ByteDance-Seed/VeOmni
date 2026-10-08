# Copyright 2026 Advanced Micro Devices, Inc. and ByteDance Ltd. and/or its affiliates
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
# Adapted from AMD-AGI/Primus (v4_sparse_mla_adapter.py, MIT License); modified for VeOmni.

"""DeepSeek-V4 sparse MQA on AMD via Triton sparse-MLA kernels.

The ROCm counterpart to ``tilelang_sparse_mla``: same compact index-list
contract, same per-head learnable sink, but MFMA Triton kernels that run on any
MFMA-capable arch instead of requiring NVIDIA SM90+. The kernels currently ship
in Primus and are imported lazily, so importing VeOmni never needs that tree.

The kernels take one flat latent pool and a token-major index list, which is a
different layout from the ``[B, S, ...]`` tensors DeepSeek-V4 attention holds.
This module owns that translation:

* ``d_qk = kv_lora_rank + rope_rank`` is the kernel's contract, with a nonzero
  rope rank. DeepSeek-V4 carries no separate rope block here, so ``ROPE_PAD``
  zero channels are appended; the kernels skip them and the backward slices
  them back off.
* the index list is rebased from per-sample to pool-global rows. Anything
  outside ``[0, kv_len)`` becomes ``-1``, as TileLang treats it.
* ``topk`` is padded to ``TOPK_ALIGN`` with ``-1``.
"""

from __future__ import annotations

import os

import torch

from ....utils.device import get_torch_device


# The kernels require ``d_qk > kv_lora_rank`` but skip the rope block in the
# single-latent form, so its contents never matter. 64 matches Primus' adapter.
ROPE_PAD = 64
# Primus' own adapter pads topk to a multiple of 64 so that backends with 64-wide
# dKV tiles stay valid. Padded slots are ``-1`` and select nothing; matching it
# keeps this path on the operand layout Primus validates.
TOPK_ALIGN = 64
# The Triton backward stages its LDS operand tiles across pipeline stages, which
# needs 160 KiB at the default schedule -- fine on CDNA4, over budget on the
# 64 KiB of CDNA3. These knobs are the backend's own; see ``_apply_lds_budget``.
_LDS_BUDGET_KNOBS = {"PRIMUS_DSA_BWD_NUM_STAGES": "1", "PRIMUS_DSA_DKV_SAFE": "1"}
_SMALL_LDS_BYTES = 64 * 1024


def _apply_lds_budget(device: torch.device) -> None:
    """Pick the small-LDS backward schedule when the device needs it.

    Measured on MI308X (gfx942, 64 KiB LDS) at S=4096 / H=64 / topk=640, the
    default schedule asks for 163840 B and ``PRIMUS_DSA_DKV_SAFE`` alone still
    asks for 73728 B; both raise ``triton.OutOfResources``. Disabling the
    pipeline staging is what brings it under budget (27.59 ms), and adding the
    narrow dKV tile is a further 12% (24.18 ms).

    ``setdefault`` rather than assignment: an explicitly exported value wins.
    Only devices that need it are touched, leaving the faster default schedule
    in place on larger LDS.
    """
    properties = get_torch_device().get_device_properties(device)
    if getattr(properties, "shared_memory_per_block", 0) > _SMALL_LDS_BYTES:
        return
    for name, value in _LDS_BUDGET_KNOBS.items():
        os.environ.setdefault(name, value)


def _load_kernels():
    try:
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import (
            sparse_mla_bwd_v4_triton,
            sparse_mla_fwd_v4_triton,
        )
    except ImportError as exc:
        raise ImportError(
            "dsa_attention_implementation='triton' could not import the DeepSeek-V4 sparse-MLA Triton "
            "kernels from Primus (primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2): "
            f"{exc}. Put the Primus source tree on PYTHONPATH."
        ) from exc
    return sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton


def _pad_rope(x: torch.Tensor) -> torch.Tensor:
    """Append ``ROPE_PAD`` zero channels to a ``[..., d]`` operand."""
    pad = x.new_zeros((*x.shape[:-1], ROPE_PAD))
    return torch.cat([x, pad], dim=-1).contiguous()


def _to_pool_indices(topk_indices: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Rebase ``[B, S, K]`` per-sample indices onto one ``B * kv_len`` pool."""
    batch = topk_indices.shape[0]
    offsets = torch.arange(batch, device=topk_indices.device, dtype=topk_indices.dtype).view(batch, 1, 1) * kv_len
    # The kernel only bounds-checks the whole pool, so a row past this sample's
    # ``kv_len`` would read the next sample's keys instead of being masked.
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    global_indices = torch.where(valid, topk_indices + offsets, torch.full_like(topk_indices, -1))
    global_indices = global_indices.reshape(-1, global_indices.shape[-1]).to(torch.int32)
    pad = (-global_indices.shape[-1]) % TOPK_ALIGN
    if pad:
        global_indices = torch.cat(
            [global_indices, global_indices.new_full((global_indices.shape[0], pad), -1)], dim=-1
        )
    return global_indices.contiguous()


class DeepSeekV4SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale):
        batch, seq_len, num_heads, head_dim = q.shape
        kv_len = kv.shape[1]
        forward, backward = _load_kernels()
        _apply_lds_budget(q.device)

        q_kernel = _pad_rope(q.reshape(batch * seq_len, num_heads, head_dim))
        kv_kernel = _pad_rope(kv.reshape(batch * kv_len, 1, head_dim))
        pool_indices = _to_pool_indices(topk_idxs, kv_len)

        o, lse = forward(
            q_kernel, kv_kernel, pool_indices, attn_sink=attn_sink, kv_lora_rank=head_dim, scale=float(sm_scale)
        )
        ctx.save_for_backward(q_kernel, kv_kernel, o, lse, pool_indices, attn_sink)
        ctx.backward_fn = backward
        ctx.shape = (batch, seq_len, num_heads, head_dim, kv_len)
        ctx.sm_scale = float(sm_scale)
        return o.reshape(batch, seq_len, num_heads, head_dim)

    @staticmethod
    def backward(ctx, do):
        q_kernel, kv_kernel, o, lse, pool_indices, attn_sink = ctx.saved_tensors
        batch, seq_len, num_heads, head_dim, kv_len = ctx.shape
        grad = do.reshape(batch * seq_len, num_heads, head_dim).contiguous()
        dq, dkv, d_attn_sink = ctx.backward_fn(
            q_kernel,
            kv_kernel,
            o,
            grad,
            pool_indices,
            lse,
            attn_sink=attn_sink,
            kv_lora_rank=head_dim,
            scale=ctx.sm_scale,
        )
        # Drop the zero rope block the operands were padded with.
        dq = dq[..., :head_dim].reshape(batch, seq_len, num_heads, head_dim)
        dkv = dkv[:, 0, :head_dim].reshape(batch, kv_len, head_dim)
        return dq, dkv, d_attn_sink, None, None


def sparse_attn_triton(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Sparse MQA over the top-k gathered KV entries.

    Args:
        q:          [B, S, H, D] bf16
        kv:         [B, S_kv, D] bf16 shared latent; K and V are the same tensor
        attn_sink:  [H] fp32
        topk_idxs:  [B, S, topk] int32 candidate rows into ``kv``, ``-1`` for
            invalid slots. Spans the sliding window and the compressed entries.
        sm_scale:   softmax scale, defaults to ``1/sqrt(D)``

    Returns:
        [B, S, H, D] bf16
    """
    # The kernels are compiled for bf16 operands. Callers run under autocast,
    # whose fp32 op policy (sum, rsqrt, ...) can silently promote an upstream
    # tensor, so reject the mismatch here instead of feeding the kernel garbage.
    if q.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise ValueError(f"DeepSeek V4 Triton sparse attention requires bfloat16 q/kv, got q={q.dtype}, kv={kv.dtype}")
    if attn_sink.dtype is not torch.float32:
        raise ValueError(f"DeepSeek V4 Triton sparse attention requires a float32 sink, got {attn_sink.dtype}")
    if q.shape[-1] != kv.shape[-1]:
        raise ValueError(
            f"DeepSeek V4 Triton sparse attention requires matching head dims, got q={q.shape[-1]}, kv={kv.shape[-1]}"
        )
    if attn_sink.shape != (q.shape[-2],):
        raise ValueError(
            "DeepSeek V4 Triton sparse attention requires one sink value per query head, got "
            f"shape={tuple(attn_sink.shape)}"
        )
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return DeepSeekV4SparseAttention.apply(
        q.contiguous(), kv.contiguous(), attn_sink.contiguous(), topk_idxs, float(sm_scale)
    )
