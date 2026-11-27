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


def dense_sparse_attn(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sparse MQA by gather-and-softmax, matching ``sparse_attn_tilelang``.

    Args:
        q: ``[B, S, H, D]``.
        kv: ``[B, S_kv, D]`` -- single MQA KV head shared across all ``H`` query heads.
        attn_sink: ``[H]`` fp32 additive logit per head that takes softmax mass
            without contributing a value.
        topk_idxs: ``[B, S, topk]`` int32 indices into ``kv``'s row axis; negative
            or out-of-range entries are masked.
        sm_scale: softmax scale, defaulting to ``D ** -0.5``.
        return_lse: also return the detached ``[B, S, H]`` fp32 base-2 log-sum-exp, sink included.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    batch, kv_len = kv.shape[:2]
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()

    rows = safe + torch.arange(batch, device=kv.device).view(-1, 1, 1) * kv_len
    gathered = kv.reshape(batch * kv_len, -1).index_select(0, rows.flatten()).view(*rows.shape, kv.shape[-1])

    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, -1)
    shift = torch.maximum(scores.max(dim=-1).values, sink)
    exp_scores = (scores - shift.unsqueeze(-1)).exp()
    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, gathered.float())
    denominator = exp_scores.sum(-1) + (sink - shift).exp()

    out = (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    if not return_lse:
        return out
    return out, ((shift + denominator.log()) / math.log(2.0)).float().detach()


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
    topk_idxs: Tensor | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Sparse MQA with HF sink-softmax math.

    Takes the ``dsa_attention`` / ``deepseek_v4`` structured contract and expands
    it into the dense mask HF's attention applies. Repeated valid candidates
    retain their per-slot softmax multiplicity, matching the TileLang row. When
    ``topk_idxs`` is supplied directly (e.g., DSpark draft attention), skips
    index construction and runs O(S * K) gather-and-softmax directly. When
    requested, LSE is detached and returned in base-2 units.
    """
    del compress_ratio, packed_sequence_slices
    kv = ori_kv if cmp_kv is None else torch.cat((ori_kv, cmp_kv), dim=1)
    if topk_idxs is not None and dropout == 0.0 and not return_attn_weights:
        return dense_sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale, return_lse=return_lse)
    if topk_idxs is None:
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
