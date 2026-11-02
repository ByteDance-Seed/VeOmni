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

"""Eager Qwen4-Exp QSA block selection — the numerical reference for the fast rows.

Mirrors the ``einsum + ReLU + topk`` selection loop of the QSA indexer patch on
the single-segment causal subset: query row ``t`` may select compressed blocks
``b < (t + 1) // compress_ratio`` (fully covered blocks only), scores are
``relu(q . k).sum(heads) / sqrt(head_dim)`` accumulated in FP32, and invisible
slots are masked with ``-1.0`` because NPU ``TopKV2`` faults on ``-inf`` rows
(ReLU scores are non-negative, so ``-1.0`` always sorts below).
"""

from __future__ import annotations

import torch
from torch import Tensor


def select_blocks(
    index_q: Tensor,
    compressed_keys: Tensor,
    compress_ratio: int,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """Return ``(indices, scores)`` of the selected compressed blocks.

    Same contract as the ``cann`` row: ``index_q`` ``[B, S, H, D]``,
    ``compressed_keys`` ``[B, N, D]`` with ``N == S // compress_ratio``;
    ``int32 [B, S, top_k]`` indices with ``-1`` padding and matching
    ``float32`` scores (``-inf`` where the index is ``-1``). ``top_k`` is
    capped at ``N``.
    """
    if index_q.ndim != 4 or compressed_keys.ndim != 3:
        raise ValueError(
            f"qsa_indexer eager takes [B,S,H,D] queries and [B,N,D] compressed keys; got {tuple(index_q.shape)} and {tuple(compressed_keys.shape)}"
        )
    batch, seq_len, num_heads, head_dim = index_q.shape
    compressed_len = compressed_keys.shape[1]
    if compressed_len != seq_len // compress_ratio:
        raise ValueError(
            f"qsa_indexer eager needs N == S // compress_ratio; got {compressed_len} for seq_len={seq_len}"
        )
    if top_k < 1:
        raise ValueError(f"qsa_indexer eager needs top_k >= 1; got {top_k}")
    width = min(top_k, compressed_len)

    scores = torch.einsum("bshd,bnd->bsnh", index_q.float(), compressed_keys.float())
    scores = torch.relu(scores).sum(dim=-1) / head_dim**0.5

    rows = torch.arange(seq_len, device=index_q.device)[:, None]
    blocks = torch.arange(compressed_len, device=index_q.device)[None, :]
    visible = blocks < (rows + 1) // compress_ratio  # [S, N]: row s may pick fully covered blocks only
    scores = scores.masked_fill(~visible, -1.0)

    top_scores, top_blocks = scores.topk(width, dim=-1)
    top_valid = visible[None].expand(batch, -1, -1).gather(-1, top_blocks)
    indices = top_blocks.to(torch.int32).masked_fill(~top_valid, -1)
    top_scores = top_scores.float().masked_fill(~top_valid, float("-inf"))

    if indices.shape[-1] < top_k:
        pad = top_k - indices.shape[-1]
        indices = torch.nn.functional.pad(indices, (0, pad), value=-1)
        top_scores = torch.nn.functional.pad(top_scores, (0, pad), value=float("-inf"))
    return indices, top_scores


def wrapper(
    index_q: Tensor,
    compressed_keys: Tensor,
    compress_ratio: int,
    top_k: int,
) -> Tensor:
    """Select compressed-KV blocks with the eager reference loop."""
    indices, _ = select_blocks(index_q, compressed_keys, compress_ratio, top_k)
    return indices
