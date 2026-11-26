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

"""Candidate lists for DeepSeek sparse attention, from the structured contract.

``dsa_attention`` describes what a query may read as a sliding window over the
full-resolution KV plus a compressed selection. Index-list kernels need that
expanded into explicit slots; the Ascend kernels take the structure itself. The
expansion lives here so each implementation does only what its kernel needs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class CompressedCandidates(NamedTuple):
    """How a compressor exposes its visible compressed entries.

    Indexer-driven compressors select an explicit per-query set and already mark
    misses with ``-1``; the heavily-compressed path instead stays a contiguous
    ``[range_start, range_end)`` causal interval. At most one of the two is set.

    ``indexer_scores`` rides alongside ``topk_indices`` rather than replacing it: the
    indexer's score at each *selected* slot, which the indexer loss needs as the
    student distribution and no index builder reads. It is ``None`` unless
    ``dsa_indexer_loss`` is on, since the eager scorer has no scores to hand over and
    the TileLang one would otherwise keep a tensor alive for nothing.
    """

    topk_indices: Tensor | None = None
    range_starts: Tensor | None = None
    range_ends: Tensor | None = None
    indexer_scores: Tensor | None = None


def _broadcast_query_ranges(bound: Tensor, batch_size: int, seq_len: int) -> Tensor:
    """Normalize a per-query range bound to ``[batch, seq_len]``."""
    if bound.ndim == 1:
        bound = bound.unsqueeze(0)
    if bound.shape[-1] != seq_len:
        raise ValueError(f"Compressed range bounds must cover {seq_len} queries; got {tuple(bound.shape)}")
    return bound.expand(batch_size, -1).to(torch.int32)


def build_sparse_attention_indices(
    *,
    position_ids: Tensor,
    sliding_window: int,
    kv_len: int,
    compressed_len: int,
    candidates: CompressedCandidates | None,
    query_offset: int = 0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Expand the structured contract into ``[B, S, window + K]`` candidate slots.

    Slots address a KV buffer that is the full-resolution rows followed by the
    compressed rows, so a compressed entry is lifted past ``kv_len``. Unusable
    slots hold ``-1``.

    Every constraint a dense mask would encode is available in closed form here,
    which is why no caller has to build one:

    * sliding causality and packed-sample isolation collapse into the single test
      ``offset <= position_ids[q]``. ``position_ids`` restarts at zero on each
      packed sample, so ``q - position_ids[q]`` is that sample's first token and
      any larger offset would reach into the previous sample.
    * compressed entries are either an explicit indexer selection that already
      marks misses with ``-1``, or a contiguous ``[range_start, range_end)``
      interval computed from the same metadata the compressor windowed with.

    ``position_ids`` are the query rows' global positions and ``seq_len`` counts
    those rows only. Under context parallelism they are one shard while the KV
    buffer stays global, so ``query_offset`` rebases a local query onto its
    absolute full-resolution row. Conflating the two is silent: the shapes stay
    self-consistent and attention simply reads the wrong rows.

    ``attention_mask`` is whatever a caller needs *on top of* sliding causality,
    such as padding. It narrows the full-resolution slots only: compressed rows
    are summaries of windows rather than tokens, so a per-token mask has no
    column for them and would reject every one of them as out of range.
    """
    if position_ids.ndim != 2:
        raise ValueError(f"Sparse attention indices need [batch, seq_len] position ids; got {position_ids.ndim} dims")

    # Transformers hands one ``[1, S]`` position row to a whole batch, so the batch
    # comes from whichever input actually carries it.
    seq_len = position_ids.shape[1]
    batch_sizes = [position_ids.shape[0]]
    if attention_mask is not None:
        batch_sizes.append(attention_mask.shape[0])
    if candidates is not None and candidates.topk_indices is not None:
        batch_sizes.append(candidates.topk_indices.shape[0])
    batch_size = max(batch_sizes)
    device = position_ids.device
    sliding_width = min(sliding_window, kv_len)
    positions = position_ids.to(torch.int32).expand(batch_size, -1)
    query_indices = torch.arange(seq_len, device=device, dtype=torch.int32) + query_offset
    window_offsets = torch.arange(sliding_width - 1, -1, -1, device=device, dtype=torch.int32)

    sliding_indices = (query_indices[None, :, None] - window_offsets[None, None, :]).expand(batch_size, -1, -1)
    sliding_indices = sliding_indices.masked_fill(window_offsets[None, None, :] > positions[:, :, None], -1)
    if attention_mask is not None:
        sliding_indices = mask_sparse_attention_indices(attention_mask, sliding_indices)

    if compressed_len == 0 or candidates is None:
        return sliding_indices.contiguous()

    if candidates.topk_indices is not None:
        compressed_indices = candidates.topk_indices.to(device=device, dtype=torch.int32)
        if compressed_indices.shape[0] not in (1, batch_size) or compressed_indices.shape[1] != seq_len:
            raise ValueError(
                "Compressed sparse indices must match the attention batch and sequence dimensions; "
                f"got {tuple(compressed_indices.shape)} for batch={batch_size}, seq_len={seq_len}"
            )
        compressed_indices = torch.where(compressed_indices >= 0, compressed_indices + kv_len, -1)
        compressed_indices = compressed_indices.expand(batch_size, -1, -1)
    elif candidates.range_starts is not None and candidates.range_ends is not None:
        starts = _broadcast_query_ranges(candidates.range_starts, batch_size, seq_len)
        ends = _broadcast_query_ranges(candidates.range_ends, batch_size, seq_len)
        entry_indices = torch.arange(compressed_len, device=device, dtype=torch.int32)
        visible = (entry_indices[None, None, :] >= starts[:, :, None]) & (
            entry_indices[None, None, :] < ends[:, :, None]
        )
        compressed_indices = torch.where(visible, entry_indices[None, None, :] + kv_len, -1)
    else:
        return sliding_indices.contiguous()

    return torch.cat((sliding_indices, compressed_indices), dim=-1).contiguous()


def mask_sparse_attention_indices(
    attention_mask: Tensor,
    sparse_indices: Tensor,
) -> Tensor:
    """Apply a dense attention mask to compact sparse candidates."""
    if attention_mask.ndim != 4 or sparse_indices.ndim != 3:
        raise ValueError(
            "Sparse attention mask/index ranks must be 4 and 3, respectively; "
            f"got {attention_mask.ndim} and {sparse_indices.ndim}"
        )
    if attention_mask.shape[-2] != sparse_indices.shape[-2]:
        raise ValueError(
            "Sparse attention mask and indices must have the same query length; "
            f"got {attention_mask.shape[-2]} and {sparse_indices.shape[-2]}"
        )

    allowed = attention_mask[:, 0] if attention_mask.dtype == torch.bool else attention_mask[:, 0] >= 0
    if allowed.shape[0] == 1 and sparse_indices.shape[0] > 1:
        allowed = allowed.expand(sparse_indices.shape[0], -1, -1)
    if allowed.shape[0] != sparse_indices.shape[0]:
        raise ValueError(
            "Sparse attention mask and indices must have compatible batch sizes; "
            f"got {allowed.shape[0]} and {sparse_indices.shape[0]}"
        )

    valid = (sparse_indices >= 0) & (sparse_indices < allowed.shape[-1])
    safe_indices = sparse_indices.clamp(min=0, max=allowed.shape[-1] - 1).to(torch.long)
    selected_valid = allowed.gather(-1, safe_indices)
    return sparse_indices.to(torch.int32).masked_fill(~valid | ~selected_valid, -1).contiguous()


__all__ = [
    "CompressedCandidates",
    "build_sparse_attention_indices",
    "mask_sparse_attention_indices",
]
