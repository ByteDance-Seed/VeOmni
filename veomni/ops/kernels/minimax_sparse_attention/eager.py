# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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

"""PyTorch MiniMax sparse index selection and attention on packed TND inputs."""

import torch
import torch.nn.functional as F

from ._common import _pack, _packed_to_padded_indices, _run_padded_attention, _unpack


def _select_blocks(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    padding: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Upstream indexer scoring on ``[B, G, S, D]`` / ``[B, 1, S, D]``; returns ``[B, G, S, topk]``."""
    batch, num_groups, seq_len, _ = index_q.shape
    positions = torch.arange(seq_len, device=index_q.device)
    num_key_blocks = -(-seq_len // block_size)
    pad = num_key_blocks * block_size - seq_len

    scores = torch.matmul(index_q.float(), index_k.float().transpose(-1, -2))
    scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
    token_future = positions[None, None, None, :] > positions[None, None, :, None]
    scores = scores.masked_fill(token_future, float("-inf"))
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    block_scores = scores.view(batch, num_groups, seq_len, num_key_blocks, block_size).amax(dim=-1)

    if local_blocks > 0:
        query_block = positions // block_size
        local = torch.arange(local_blocks, device=index_q.device)
        local_idx = (query_block[:, None] - local[None, :]).clamp(min=0)
        block_scores.scatter_(-1, local_idx.expand(batch, num_groups, -1, -1), float("inf"))

    topk = min(topk_blocks, num_key_blocks)
    topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
    block_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
    return block_indices.masked_fill(padding[:, None, :, None], -1)


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep: torch.Tensor,
    *,
    scale: float,
    dropout_p: float,
) -> torch.Tensor:
    num_groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(num_groups, dim=1)
    value = value.repeat_interleave(num_groups, dim=1)
    attention_mask = torch.zeros(keep.shape, dtype=query.dtype, device=query.device).masked_fill(
        ~keep, torch.finfo(query.dtype).min
    )

    attention_weights = torch.matmul(query, key.transpose(2, 3)) * scale
    attention_weights = attention_weights + attention_mask
    attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout_p > 0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)
    return torch.matmul(attention_weights, value).transpose(1, 2).contiguous()


def minimax_sparse_indexer_eager(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select ``[T, G, topk_blocks]`` int32 key blocks per query and KV group; see the package docstring."""
    if index_k.shape[1] != 1:
        raise ValueError(f"MiniMax sparse indexer expects a single index key head, got {index_k.shape[1]}.")

    indices, padding = _packed_to_padded_indices(cu_seqlens, max_seqlen, index_q.shape[0], index_q.device)
    padded_q = _unpack(index_q, indices, padding).transpose(1, 2)
    padded_k = _unpack(index_k, indices, padding).transpose(1, 2)
    block_indices = _select_blocks(
        padded_q,
        padded_k,
        padding,
        block_size=block_size,
        topk_blocks=topk_blocks,
        local_blocks=local_blocks,
    )
    block_indices = _pack(block_indices.transpose(1, 2), indices)
    # Short sequences have fewer key blocks than slots; the extra slots are empty.
    block_indices = F.pad(block_indices, (0, topk_blocks - block_indices.shape[-1]), value=-1)
    return block_indices.to(torch.int32)


def minimax_sparse_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Apply PyTorch attention to packed Q/K/V using per-group block indices."""
    return _run_padded_attention(
        q,
        k,
        v,
        block_indices,
        cu_seqlens,
        max_seqlen,
        block_size=block_size,
        scale=scale,
        dropout_p=dropout_p,
        attend=_eager_attention,
        via_bfloat16=False,
    )
