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
# See the License for the specific language governing limitations
# under the License.

"""GLM-DSA sparse MLA eager.

Adapted from HuggingFace ``GlmMoeDsaAttention.forward`` eager path: build a
topk additive mask, then run standard attention. Packed q/k here is the
FlashMLA layout used by the fused row (``q = cat(q_nope, q_pe)``,
``kv = cat(kv_cache, k_pe)``), value dim stays ``kv_cache``.

https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def wrapper(
    q_pe: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    q_nope_absorbed: Tensor,
    topk_indices: Tensor,
    *,
    softmax_scale: float | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Official GLM eager attention on the FlashMLA packed face.

    ``q_pe`` / ``q_nope_absorbed`` are ``[B, S, H, D]``. ``k_pe`` and
    ``kv_cache`` are MQA ``[B, S_kv, 1, D]``.
    ``attention_mask`` is the official additive causal / padding mask,
    broadcastable to ``[B, 1, S, T]``.
    """
    query = torch.cat((q_nope_absorbed, q_pe), dim=-1)
    key = torch.cat((kv_cache.squeeze(2), k_pe.squeeze(2)), dim=-1)
    value = kv_cache.squeeze(2)
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    batch, q_len, heads, _ = query.shape
    kv_len = key.shape[1]
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    safe = topk_indices.clamp(0, max(kv_len - 1, 0)).long()
    # scatter_add so -1 sentinels (clamped to 0) cannot overwrite a real keep.
    keep = torch.zeros(batch, q_len, kv_len, dtype=torch.int32, device=query.device)
    keep.scatter_add_(-1, safe, valid.to(keep.dtype))
    keep = keep > 0
    index_mask = torch.where(
        keep.unsqueeze(1),
        torch.zeros((), device=query.device, dtype=query.dtype),
        torch.full((), float("-inf"), device=query.device, dtype=query.dtype),
    )
    query_h = query.transpose(1, 2)
    key_h = key.unsqueeze(1).expand(-1, heads, -1, -1)
    value_h = value.unsqueeze(1).expand(-1, heads, -1, -1)
    attn_weights = torch.matmul(query_h, key_h.transpose(2, 3)) * scale + index_mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., :kv_len]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_h.dtype)
    out = torch.matmul(attn_weights, value_h)
    return out.transpose(1, 2).contiguous()
