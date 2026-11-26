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

"""DeepSeek-V4 DSA attention eager.

Adapted from HuggingFace ``eager_attention_forward`` (transformers v5.9.0):
matmul scores, add mask, concat per-head sink, shift by max, softmax, drop
the sink column, then ``scores @ V``. The candidate list becomes the additive
mask. KV is MQA (one shared head).

https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ...sparse_indices import CompressedCandidates, build_sparse_attention_indices


def _topk_to_additive_mask(topk_idxs: Tensor, kv_len: int, dtype: torch.dtype) -> Tensor:
    """Dense mask preserving the candidate-list multiplicity of each key."""
    batch, q_len, _ = topk_idxs.shape
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()
    counts = torch.zeros(batch, q_len, kv_len, dtype=torch.int32, device=topk_idxs.device)
    counts.scatter_add_(-1, safe, valid.to(counts.dtype))
    # Repeating a candidate n times in the gathered TileLang softmax is
    # equivalent to adding log(n) to that key's dense logit.
    multiplicity_bias = counts.to(dtype).log()
    min_value = torch.finfo(dtype).min
    return torch.where(counts.unsqueeze(1) > 0, multiplicity_bias.unsqueeze(1), min_value)


def wrapper(
    q: Tensor,
    ori_kv: Tensor,
    cmp_kv: Tensor | None,
    attn_sink: Tensor,
    position_ids: Tensor,
    *,
    sliding_window: int,
    compress_ratio: int,
    candidates: CompressedCandidates | None = None,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    query_offset: int = 0,
    attention_mask: Tensor | None = None,
    sm_scale: float | None = None,
    return_lse: bool = False,
    dropout: float = 0.0,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    """Sparse MQA with HF sink-softmax math.

    Takes the ``dsa_attention`` / ``deepseek_v4`` structured contract and expands
    it into the dense mask HF's attention applies. Repeated valid candidates
    retain their per-slot softmax multiplicity, matching the TileLang row. When
    requested, LSE is detached and returned in base-2 units.
    """
    del compress_ratio, packed_sequence_slices
    kv = ori_kv if cmp_kv is None else torch.cat((ori_kv, cmp_kv), dim=1)
    topk_idxs = build_sparse_attention_indices(
        position_ids=position_ids,
        sliding_window=sliding_window,
        kv_len=ori_kv.shape[1],
        compressed_len=0 if cmp_kv is None else cmp_kv.shape[1],
        candidates=candidates,
        query_offset=query_offset,
        attention_mask=attention_mask,
    )
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
    if dropout > 0.0:
        scores = F.dropout(scores, p=dropout, training=True)
    attn_weights_out = scores.to(value_states.dtype)
    attn_output = torch.matmul(attn_weights_out, value_states)
    out = attn_output.transpose(1, 2).contiguous()
    extras: tuple[Tensor, ...] = ()
    if return_lse:
        lse = (
            torch.logsumexp(torch.cat([attn_weights.float(), sinks.float()], dim=-1), dim=-1) * math.log2(math.e)
        ).transpose(1, 2)
        extras += (lse.detach(),)
    if return_attn_weights:
        extras += (attn_weights_out,)
    if extras:
        return (out, *extras)
    return out
