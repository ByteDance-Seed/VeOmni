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

"""DeepSeek-V4 DSA attention eager.

Adapted from HuggingFace ``eager_attention_forward`` (transformers v5.9.0):
matmul scores, add mask, concat per-head sink, shift by max, softmax, drop
the sink column, then ``scores @ V``. Top-k indices become the additive
mask. KV is MQA (one shared head).

https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _topk_to_additive_mask(topk_idxs: Tensor, kv_len: int, dtype: torch.dtype) -> Tensor:
    """HF additive mask: 0 at selected keys, ``finfo.min`` elsewhere."""
    batch, q_len, _ = topk_idxs.shape
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()
    # scatter_add so -1 sentinels (clamped to 0) cannot overwrite a real keep.
    keep = torch.zeros(batch, q_len, kv_len, dtype=torch.int32, device=topk_idxs.device)
    keep.scatter_add_(-1, safe, valid.to(keep.dtype))
    keep = keep > 0
    min_value = torch.finfo(dtype).min
    return torch.where(keep.unsqueeze(1), torch.zeros((), device=topk_idxs.device, dtype=dtype), min_value)


def wrapper(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sparse MQA with HF sink-softmax math.

    ``q`` is ``[B, S, H, D]``, ``kv`` is ``[B, S_kv, D]``, ``attn_sink`` is
    ``[H]``, ``topk_idxs`` is ``[B, S, topk]``.
    """
    scale = q.shape[-1] ** -0.5 if sm_scale is None else sm_scale
    query = q.transpose(1, 2).contiguous()
    key_states = kv.unsqueeze(1).expand(-1, q.shape[2], -1, -1).contiguous()
    value_states = key_states
    mask = _topk_to_additive_mask(topk_idxs, kv.shape[1], query.dtype)

    # Copied from eager_attention_forward (transformers v5.9.0).
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scale
    attn_weights = attn_weights + mask
    sinks = attn_sink.to(dtype=query.dtype).reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores.to(value_states.dtype), value_states)
    out = attn_output.transpose(1, 2).contiguous()
    if not return_lse:
        return out
    lse = torch.logsumexp(torch.cat([attn_weights.float(), sinks.float()], dim=-1), dim=-1).transpose(1, 2)
    return out, lse.detach()
