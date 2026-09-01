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

"""DeepSeek-V4 lighting indexer eager.

Adapted from HuggingFace ``DeepseekV4Indexer.forward`` score / top-k
(transformers v5.9.0). The official path is

    scores = ReLU(q @ k^T) * softmax_scale
    index_scores = (scores * weights).sum(heads)

This wrapper uses the same matmul / ReLU / weighted sum / causal top-k.
``softmax_scale`` is folded into ``weights`` so the call face stays
``(index_q, index_k, weights, compress_ratio, topk, ...)``. Callers that
match HF should pass ``weights_proj * weights_scaling * softmax_scale``.

https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _extract_topk_scores(logits: Tensor, topk_indices: Tensor) -> Tensor:
    """Gather scores at ``topk_indices``. Invalid ids become ``-inf``."""
    valid = (topk_indices >= 0) & (topk_indices < logits.shape[-1])
    safe = topk_indices.clamp(min=0, max=max(logits.shape[-1] - 1, 0)).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe)
    return torch.where(valid, scores, float("-inf"))


def wrapper(
    index_q: Tensor,
    index_k: Tensor,
    weights: Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: Tensor | None = None,
    cu_seqlen_ks: Tensor | None = None,
    cu_seqlen_ke: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """HF indexer scores on the kernel face.

    ``index_q`` is ``[S, B, H, D]``, ``index_k`` is ``[S_kv, B, D]``,
    ``weights`` is ``[S, B, H]``. Returns ``(index_score, topk_indices)``
    with shapes ``[B, S, topk]``.
    """
    if (cu_seqlen_ks is None) != (cu_seqlen_ke is None):
        raise ValueError("cu_seqlen_ks and cu_seqlen_ke must be provided together")

    # HF layout: q [B, S, H, D], compressed_kv [B, T, D], weights [B, S, H]
    q = index_q.permute(1, 0, 2, 3).contiguous()
    compressed_kv = index_k.transpose(0, 1).contiguous()
    eager_weights = weights.permute(1, 0, 2).contiguous()
    batch, seq_len, _heads, _dim = q.shape
    compressed_len = compressed_kv.shape[1]

    # Copied from DeepseekV4Indexer.forward (transformers v5.9.0):
    # scores = matmul(q.float(), compressed_kv.T.unsqueeze(1))
    # scores = relu(scores) * softmax_scale
    # index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores)
    index_scores = (scores * eager_weights.float().unsqueeze(-1)).sum(dim=2)

    if cu_seqlen_ks is None:
        position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch, -1)
        causal_threshold = (position_ids + 1) // compress_ratio
        if compressed_len > 0:
            entry_indices = torch.arange(compressed_len, device=index_scores.device)
            future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
            index_scores = index_scores.masked_fill(future_mask, float("-inf"))
    elif cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            "Packed indexer ranges must have shape "
            f"({seq_len},), got {tuple(cu_seqlen_ks.shape)} and {tuple(cu_seqlen_ke.shape)}"
        )
    else:
        entries = torch.arange(compressed_len, device=q.device)
        valid_ranges = (entries >= cu_seqlen_ks[:, None]) & (entries < cu_seqlen_ke[:, None])
        index_scores = index_scores.masked_fill(~valid_ranges.unsqueeze(0), float("-inf"))
        causal_threshold = None

    if topk_indices is not None:
        return _extract_topk_scores(index_scores, topk_indices), topk_indices

    top_k = min(topk, max(compressed_len, 1))
    if compressed_len == 0:
        top_k_indices = index_scores.topk(top_k, dim=-1).indices.to(torch.int32)
        return _extract_topk_scores(index_scores, top_k_indices), top_k_indices

    top_k_indices = index_scores.topk(top_k, dim=-1).indices
    if causal_threshold is not None:
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
    else:
        invalid = (top_k_indices < cu_seqlen_ks.view(1, -1, 1)) | (top_k_indices >= cu_seqlen_ke.view(1, -1, 1))
    top_k_indices = torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices).to(torch.int32)
    return _extract_topk_scores(index_scores, top_k_indices), top_k_indices
