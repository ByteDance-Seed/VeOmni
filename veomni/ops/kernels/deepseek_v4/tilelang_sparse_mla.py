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
# Adapted from radixark/miles; modified for VeOmni.

import torch

from . import tilelang_sparse_mla_bwd as sparse_mla_bwd
from . import tilelang_sparse_mla_fwd as sparse_mla_fwd


class DeepSeekV4SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None):
        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q, kv, attn_sink, o, do.contiguous(), topk_idxs, lse, sm_scale=sm_scale
        )

        return dq, dkv, d_attn_sink, None, None


def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Sparse MQA over the top-k gathered KV entries.

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        attn_sink: [H] fp32
        topk_idxs: [B, S, topk] int32
        sm_scale:  softmax scale, defaults to ``1/sqrt(D)``

    Returns:
        [B, S, H, D] bf16
    """
    # The kernels are compiled for bf16 operands. Callers run under autocast,
    # whose fp32 op policy (sum, rsqrt, ...) can silently promote an upstream
    # tensor, so reject the mismatch here instead of feeding the kernel garbage.
    if q.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise ValueError(
            f"DeepSeek V4 TileLang sparse attention requires bfloat16 q/kv, got q={q.dtype}, kv={kv.dtype}"
        )
    if attn_sink.dtype is not torch.float32:
        raise ValueError(f"DeepSeek V4 TileLang sparse attention requires a float32 sink, got {attn_sink.dtype}")
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)
