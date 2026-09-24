# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compact global-selection helpers for Qwen4-Exp QSA under Ulysses."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from ....distributed.context_parallel.dsa_cp import all_gather_compressed_rows
from ....distributed.sequence_parallel import gather_outputs


_MAX_SCORE_ELEMENTS = 16 * 1024 * 1024


def _packed_segments(
    batch_size: int,
    seq_len: int,
    cu_seq_lens_q: torch.Tensor | None,
) -> list[tuple[int, int, int]]:
    """Return ``(batch, start, end)`` ranges in batch-local coordinates."""
    if cu_seq_lens_q is None:
        return [(batch_idx, 0, seq_len) for batch_idx in range(batch_size)]

    boundaries = [int(value) for value in cu_seq_lens_q.tolist()]
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != batch_size * seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA requires cu_seq_lens_q to cover the complete padded batch; "
            f"got endpoints {boundaries[:1]}..{boundaries[-1:]}, expected 0..{batch_size * seq_len}."
        )
    segments = []
    for flat_start, flat_end in zip(boundaries, boundaries[1:]):
        if flat_end <= flat_start:
            raise ValueError("Qwen4-Exp compact QSA requires strictly increasing cu_seq_lens_q.")
        batch_idx = flat_start // seq_len
        if (flat_end - 1) // seq_len != batch_idx:
            raise ValueError("Qwen4-Exp compact QSA packed samples may not cross the batch dimension.")
        segments.append((batch_idx, flat_start - batch_idx * seq_len, flat_end - batch_idx * seq_len))
    return segments


def _segment_ids_and_starts(
    batch_size: int,
    seq_len: int,
    segments: list[tuple[int, int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    segment_ids = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    segment_starts = torch.empty_like(segment_ids)
    for segment_id, (batch_idx, start, end) in enumerate(segments):
        segment_ids[batch_idx, start:end] = segment_id
        segment_starts[batch_idx, start:end] = start
    return segment_ids, segment_starts


def _right_halo(raw_keys: torch.Tensor, halo: int, group, rank: int, world_size: int) -> torch.Tensor:
    """Gather short prefixes and return the next rank's differentiable halo."""
    if halo == 0 or world_size == 1:
        return raw_keys[:, :0]
    prefixes = gather_outputs(raw_keys[:, :halo].contiguous(), gather_dim=1, group=group)
    if rank + 1 == world_size:
        return prefixes[:, :halo] * 0
    start = (rank + 1) * halo
    return prefixes[:, start : start + halo]


def compact_qsa_select(
    indexer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seq_lens_q: torch.Tensor | None,
    *,
    group,
    rank: int,
    world_size: int,
    apply_rotary_pos_emb: Callable,
) -> torch.Tensor:
    """Return local-query/global-token QSA selections as ``[B, L, K]`` int32."""
    batch_size, local_seq_len, _ = hidden_states.shape
    global_seq_len = local_seq_len * world_size
    if indexer.index_kv_heads != 1:
        raise ValueError(
            f"Qwen4-Exp compact QSA currently requires one indexer KV head; got {indexer.index_kv_heads}."
        )
    if indexer.compress_ratio > local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA needs each Ulysses shard to be at least one compression block wide; "
            f"got local_seq_len={local_seq_len}, compress_ratio={indexer.compress_ratio}."
        )

    qk = indexer.index_qk_proj(hidden_states)
    q_width = indexer.index_n_heads * indexer.index_head_dim
    q, raw_keys = torch.split(qk, [q_width, indexer.index_head_dim], dim=-1)
    q = indexer.q_layernorm(q.reshape(batch_size, local_seq_len, indexer.index_n_heads, indexer.index_head_dim))
    full_cos, full_sin = position_embeddings
    local_start = rank * local_seq_len
    local_cos = full_cos[:, local_start : local_start + local_seq_len]
    local_sin = full_sin[:, local_start : local_start + local_seq_len]
    q = apply_rotary_pos_emb(q, cos=local_cos, sin=local_sin, unsqueeze_dim=2)

    segments = _packed_segments(batch_size, global_seq_len, cu_seq_lens_q)
    segment_ids, segment_starts = _segment_ids_and_starts(batch_size, global_seq_len, segments, hidden_states.device)
    blocks = [
        (segment_id, batch_idx, start)
        for segment_id, (batch_idx, segment_start, segment_end) in enumerate(segments)
        for start in range(segment_start, segment_end - indexer.compress_ratio + 1, indexer.compress_ratio)
    ]
    owned_blocks = [block for block in blocks if block[2] // local_seq_len == rank]
    counts = torch.tensor(
        [sum(block_start // local_seq_len == owner for _, _, block_start in blocks) for owner in range(world_size)],
        dtype=torch.long,
        device=hidden_states.device,
    )

    halo = indexer.compress_ratio - 1
    extended_keys = torch.cat((raw_keys, _right_halo(raw_keys, halo, group, rank, world_size)), dim=1)
    if owned_blocks:
        local_batch_ids = torch.tensor(
            [batch_idx for _, batch_idx, _ in owned_blocks], dtype=torch.long, device=hidden_states.device
        )
        local_block_starts = torch.tensor(
            [start for _, _, start in owned_blocks], dtype=torch.long, device=hidden_states.device
        )
        local_offsets = local_block_starts - local_start
        window_indices = (
            local_offsets[:, None] + torch.arange(indexer.compress_ratio, device=hidden_states.device)[None]
        )
        pooled_keys = extended_keys[local_batch_ids[:, None], window_indices].float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = indexer.k_layernorm(pooled_keys)
        pooled_keys = apply_rotary_pos_emb(
            pooled_keys.unsqueeze(1),
            cos=full_cos[local_batch_ids, local_block_starts],
            sin=full_sin[local_batch_ids, local_block_starts],
        ).squeeze(1)
        pooled_keys = pooled_keys.unsqueeze(0)
    else:
        # Keep both the projection and RMSNorm parameters in the autograd graph
        # on ranks that own no complete block.
        pooled_keys = indexer.k_layernorm(extended_keys[:, :0]).reshape(1, 0, indexer.index_head_dim)

    if world_size > 1:
        pooled_keys = all_gather_compressed_rows(pooled_keys, counts, group)
    gathered_blocks = [block for owner in range(world_size) for block in blocks if block[2] // local_seq_len == owner]
    output_width = indexer.token_budget + indexer.compress_ratio - 1
    query_positions = local_start + torch.arange(local_seq_len, device=hidden_states.device)
    local_segment_ids = segment_ids[:, local_start : local_start + local_seq_len]
    local_segment_starts = segment_starts[:, local_start : local_start + local_seq_len]

    if gathered_blocks:
        block_segment_ids = torch.tensor(
            [segment_id for segment_id, _, _ in gathered_blocks], dtype=torch.long, device=hidden_states.device
        )
        block_starts = torch.tensor(
            [start for _, _, start in gathered_blocks], dtype=torch.long, device=hidden_states.device
        )
        top_count = min(indexer.block_topk, len(gathered_blocks))
        block_offsets = torch.arange(indexer.compress_ratio, device=hidden_states.device)
        elements_per_query = max(1, batch_size * len(gathered_blocks) * indexer.index_n_heads)
        query_chunk_size = max(1, min(local_seq_len, _MAX_SCORE_ELEMENTS // elements_per_query))
        selected_chunks = []
        for query_start in range(0, local_seq_len, query_chunk_size):
            query_end = min(query_start + query_chunk_size, local_seq_len)
            scores = torch.einsum(
                "blhd,nd->blnh",
                q[:, query_start:query_end].float(),
                pooled_keys[0].float(),
            )
            scores = torch.relu(scores).sum(dim=-1) / indexer.index_head_dim**0.5
            visible = (block_segment_ids[None, None] == local_segment_ids[:, query_start:query_end, None]) & (
                block_starts[None, None] + indexer.compress_ratio - 1
                <= query_positions[None, query_start:query_end, None]
            )
            scores = scores.masked_fill(~visible, float("-inf"))
            top_blocks = scores.topk(top_count, dim=-1).indices
            top_valid = visible.gather(-1, top_blocks)
            selected_starts = block_starts[top_blocks]
            selected_chunk = selected_starts[..., None] + block_offsets
            selected_chunks.append(selected_chunk.masked_fill(~top_valid[..., None], -1).flatten(-2))
        selected_blocks = torch.cat(selected_chunks, dim=1)
    else:
        selected_blocks = torch.empty(batch_size, local_seq_len, 0, dtype=torch.long, device=hidden_states.device)

    visible_count = query_positions[None] - local_segment_starts + 1
    tail_count = torch.remainder(visible_count, indexer.compress_ratio)
    tail_start = (
        local_segment_starts
        + torch.div(visible_count, indexer.compress_ratio, rounding_mode="floor") * indexer.compress_ratio
    )
    tail_offsets = torch.arange(indexer.compress_ratio - 1, device=hidden_states.device)
    tails = tail_start[..., None] + tail_offsets
    tails = tails.masked_fill(tail_offsets[None, None] >= tail_count[..., None], -1)

    selected = torch.cat((selected_blocks, tails), dim=-1)
    if selected.shape[-1] < output_width:
        selected = torch.nn.functional.pad(selected, (0, output_width - selected.shape[-1]), value=-1)
    return selected[..., :output_width].to(torch.int32).contiguous()


def gather_qsa_selected_indices(local_indices: torch.Tensor, group, world_size: int) -> torch.Tensor:
    """Assemble local query selections into global query order."""
    if world_size == 1:
        return local_indices
    gathered = [torch.empty_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered, local_indices, group=group)
    return torch.cat(gathered, dim=1)


__all__ = ["compact_qsa_select", "gather_qsa_selected_indices"]
