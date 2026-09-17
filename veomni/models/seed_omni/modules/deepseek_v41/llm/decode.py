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

"""The decode-step half of V4.1 attention: one query against cached KV.

``attention.py`` is prefill: it builds every layer's KV from the hidden states in
front of it and throws the result away. Decoding needs the opposite -- one new
query per step against KV that was computed on earlier steps and must survive
them -- so this file holds the state prefill deliberately does not keep.

What makes that tractable is a property of the kernel rather than of the model:
``sparse_attn_tilelang`` takes ``kv`` as an unordered *pool* of rows and
``topk_idxs`` as indices into it. Nothing requires ``kv[0:S]`` to be the current
forward's own tokens. A decode step therefore reuses the prefill call shape
exactly, with the pool swapped for ``cat([window_cache, compress_cache])``; only
the index arithmetic differs, and it is the same arithmetic evaluated at one
query instead of ``S``.

Attention here runs dense rather than through the sparse kernel, which is a
deliberate difference from prefill and not a fallback of convenience:

* at ``S == 1`` the gathered pool is ``[1, 1, window + index_topk, head_dim]`` --
  640 rows at the production shape. Materializing it costs less than launching a
  kernel tiled for thousands of query rows;
* the kernel needs SM90+ and TileLang. Decode is how the model is *inspected* --
  a chat transcript, a logit dump -- and tying that to the training accelerator
  means the cheapest question about a checkpoint needs the scarcest hardware;
* it makes the decode path checkable against the kernel rather than only against
  itself, which is what ``tests/seed_omni/test_deepseek_v41_decode.py`` does.

``dense_sparse_attn`` is therefore held to being the kernel's equal, not its
approximation, and the sink is the part worth naming: it enters the denominator
and contributes no value, so dropping it leaves a distribution that still sums
to one and is wrong by whatever mass the sink held.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def dense_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """Sparse MQA by gather-and-softmax, matching ``sparse_attn_tilelang``.

    Args:
        q: ``[B, S, H, D]``.
        kv: ``[B, S_kv, D]`` -- one KV head shared by all ``H`` query heads, and
            the same rows serve as keys and as values.
        attn_sink: ``[H]`` fp32, an additive logit per head that takes softmax
            mass without contributing a value.
        topk_idxs: ``[B, S, topk]`` int32 into ``kv``'s row axis. Negative or
            out-of-range entries are masked, which is how the window's
            not-yet-filled slots and the indexer's misses both arrive.
        sm_scale: softmax scale, defaulting to ``D ** -0.5``.
        return_lse: also return the ``[B, S, H]`` fp32 log-sum-exp, sink
            included. **Base two**, matching the kernel -- see below.

    Returns:
        ``[B, S, H, D]`` in ``q``'s dtype, and the LSE when asked for it.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    kv_len = kv.shape[1]
    # Out-of-range is masked rather than rejected: the kernel treats any index it
    # cannot read as a miss, and a decode step legitimately produces them from a
    # window that is not yet full.
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()

    gathered = kv[:, None, :, :].expand(-1, q.shape[1], -1, -1)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, kv.shape[-1]))

    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # The sink is part of the softmax, so it belongs in the max the exponentials
    # are shifted by. Shifting by the scores' max alone is stable for the output
    # -- 0/inf is 0 -- but sends the LSE to +inf on a fully masked query, where
    # that max is -inf: the shift then becomes exp(sink + 1e30). Folding the sink
    # in leaves such a query with denominator 1 and LSE exactly the sink, which
    # is what a query attending to nothing but the sink means.
    shift = torch.maximum(scores.max(dim=-1).values, attn_sink.view(1, 1, -1))
    exp_scores = (scores - shift.unsqueeze(-1)).exp()
    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, gathered.float())
    denominator = exp_scores.sum(-1) + (attn_sink.view(1, 1, -1) - shift).exp()

    out = (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    if not return_lse:
        return out
    # The kernel exponentiates with exp2 and returns a base-two LSE; its wrapper
    # documents the tensor only as "the log-sum-exp". Returning natural log here
    # would be wrong by log2(e) = 1.4427 -- and wrong in a way that survives every
    # check that does not compare against the kernel, since the consumer
    # (``sparse_mqa_target_fwd``, for the indexer teacher) takes the LSE as a
    # renormalizer and a uniformly scaled one still yields a distribution.
    return out, ((shift + denominator.log()) / math.log(2.0)).float()


__all__ = ["dense_sparse_attn"]
