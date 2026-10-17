# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compact global-selection helpers for Qwen4-Exp QSA under Ulysses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ....distributed.context_parallel.dsa_cp import all_gather_compressed_rows
from ....distributed.sequence_parallel import gather_outputs


_MAX_SCORE_ELEMENTS = 16 * 1024 * 1024


def _packed_segments(
    batch_size: int,
    seq_len: int,
    cu_seq_lens_q: torch.Tensor | None,
    packed_seq_lens: tuple[int, ...] | None = None,
) -> list[tuple[int, int, int]]:
    """Return ``(batch, start, end)`` ranges in batch-local coordinates."""
    if packed_seq_lens is not None:
        boundaries = [0]
        for length in packed_seq_lens:
            boundaries.append(boundaries[-1] + int(length))
    elif cu_seq_lens_q is None:
        return [(batch_idx, 0, seq_len) for batch_idx in range(batch_size)]
    else:
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


@dataclass(frozen=True)
class PackedQSALayout:
    """Integer-only selection metadata; never holds projected keys or autograd state."""

    counts: tuple[int, ...]
    local_batch_ids: torch.Tensor
    local_block_starts: torch.Tensor
    window_indices: torch.Tensor
    block_segment_ids: torch.Tensor
    block_batch_ids: torch.Tensor
    block_owners: torch.Tensor
    block_starts: torch.Tensor
    gather_keep_indices: torch.Tensor
    query_positions: torch.Tensor
    local_segment_ids: torch.Tensor
    local_segment_starts: torch.Tensor
    block_offsets: torch.Tensor
    tails: torch.Tensor


def build_packed_qsa_layout(segments, batch_size, global_seq_len, compress_ratio, rank, world_size, device):
    """Build stable owner-major indices on the host, then upload each integer tensor once."""
    local_seq_len = global_seq_len // world_size
    starts, batches, sample_ids = [], [], []
    for segment_id, (batch_idx, start, end) in enumerate(segments):
        block_starts = torch.arange(
            start, max(start, end - compress_ratio + 1), compress_ratio, dtype=torch.long, device="cpu"
        )
        starts.append(block_starts)
        batches.append(torch.full_like(block_starts, batch_idx))
        sample_ids.append(torch.full_like(block_starts, segment_id))
    starts = torch.cat(starts)
    batches = torch.cat(batches)
    sample_ids = torch.cat(sample_ids)
    owners = torch.div(starts, local_seq_len, rounding_mode="floor")
    counts_tensor = torch.bincount(owners, minlength=world_size)
    counts = tuple(counts_tensor.tolist())  # CPU integer metadata, no accelerator readback.
    # Preserve segment/batch-major order within each owner, including equal starts.
    order = torch.argsort(owners, stable=True)
    starts, batches, sample_ids = starts[order], batches[order], sample_ids[order]
    first = sum(counts[:rank])
    last = first + counts[rank]
    local_start = rank * local_seq_len
    local_batch_ids = batches[first:last].to(device)
    local_block_starts = starts[first:last].to(device)
    block_offsets = torch.arange(compress_ratio, device=device)
    window_indices = local_block_starts[:, None] - local_start + block_offsets[None]
    n_max = max(counts, default=0)
    rows = torch.arange(n_max, device="cpu")
    keep = torch.arange(world_size, device="cpu")[:, None] * n_max + rows[None]
    keep = keep[rows[None] < counts_tensor[:, None]].to(device)
    segment_ids, segment_starts = _segment_ids_and_starts(batch_size, global_seq_len, segments, device)
    local_segment_ids = segment_ids[:, local_start : local_start + local_seq_len].clone()
    local_segment_starts = segment_starts[:, local_start : local_start + local_seq_len].clone()
    query_positions = local_start + torch.arange(local_seq_len, device=device)
    visible_count = query_positions[None] - local_segment_starts + 1
    tail_count = torch.remainder(visible_count, compress_ratio)
    tail_start = (
        local_segment_starts + torch.div(visible_count, compress_ratio, rounding_mode="floor") * compress_ratio
    )
    tail_offsets = torch.arange(compress_ratio - 1, device=device)
    tails = tail_start[..., None] + tail_offsets
    tails = tails.masked_fill(tail_offsets[None, None] >= tail_count[..., None], -1)
    return PackedQSALayout(
        counts,
        local_batch_ids,
        local_block_starts,
        window_indices,
        sample_ids.to(device),
        batches.to(device),
        owners[order].to(device),
        starts.to(device),
        keep,
        query_positions,
        local_segment_ids,
        local_segment_starts,
        block_offsets,
        tails,
    )


def get_packed_qsa_layout(
    batch_size,
    global_seq_len,
    cu_seq_lens_q,
    compress_ratio,
    rank,
    world_size,
    device,
    *,
    packed_seq_lens=None,
    layout_cache=None,
):
    """A content-keyed, per-forward cache; callers must pass validated host lengths."""
    segments = _packed_segments(batch_size, global_seq_len, cu_seq_lens_q, packed_seq_lens)
    device = torch.device(device)
    key = (tuple(segments), batch_size, global_seq_len, compress_ratio, rank, world_size, device)
    if layout_cache is not None and key in layout_cache:
        return layout_cache[key]
    layout = build_packed_qsa_layout(segments, batch_size, global_seq_len, compress_ratio, rank, world_size, device)
    if layout_cache is not None:
        layout_cache[key] = layout
    return layout


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
    packed_seq_lens: tuple[int, ...] | None = None,
    layout_cache: dict | None = None,
) -> torch.Tensor:
    """Return local-query/global-token QSA selections as ``[B, L, K]`` int32."""
    batch_size, local_seq_len, _ = hidden_states.shape
    global_seq_len = local_seq_len * world_size
    if indexer.index_kv_heads != 1:
        raise ValueError(
            f"Qwen4-Exp compact QSA currently requires one indexer KV head; got {indexer.index_kv_heads}."
        )
    if world_size > 1 and indexer.compress_ratio > local_seq_len:
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

    layout = get_packed_qsa_layout(
        batch_size,
        global_seq_len,
        cu_seq_lens_q,
        indexer.compress_ratio,
        rank,
        world_size,
        hidden_states.device,
        packed_seq_lens=packed_seq_lens,
        layout_cache=layout_cache,
    )

    halo = indexer.compress_ratio - 1
    extended_keys = torch.cat((raw_keys, _right_halo(raw_keys, halo, group, rank, world_size)), dim=1)
    if layout.counts[rank]:
        local_batch_ids = layout.local_batch_ids
        local_block_starts = layout.local_block_starts
        window_indices = layout.window_indices
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
        pooled_keys = all_gather_compressed_rows(
            pooled_keys, layout.counts, group, keep_indices=layout.gather_keep_indices
        )
    block_count = layout.block_starts.numel()
    output_width = indexer.token_budget + indexer.compress_ratio - 1
    query_positions = layout.query_positions
    local_segment_ids = layout.local_segment_ids

    if block_count:
        block_segment_ids = layout.block_segment_ids
        block_starts = layout.block_starts
        top_count = min(indexer.block_topk, block_count)
        block_offsets = layout.block_offsets
        elements_per_query = max(1, batch_size * block_count * indexer.index_n_heads)
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

    tails = layout.tails

    selected = torch.cat((selected_blocks, tails), dim=-1)
    if selected.shape[-1] < output_width:
        selected = torch.nn.functional.pad(selected, (0, output_width - selected.shape[-1]), value=-1)
    return selected[..., :output_width].to(torch.int32).contiguous()


def explicit_mask_qsa_select(indexer, hidden_states, position_embeddings, attention_mask, apply_rotary_pos_emb):
    """Match HF visible-token block grouping for explicitly masked, non-CP calls.

    Explicit masks can contain holes, so their compression blocks are groups
    of visible tokens rather than fixed contiguous windows. Keep this rare
    compatibility path separate from the bounded packed-training fast path.
    """
    batch_size, seq_len, _ = hidden_states.shape
    if attention_mask.ndim != 4 or attention_mask.shape[1] != 1:
        raise ValueError("Explicit QSA masks require shape [B,1,Q,K] with a shared head mask")
    mask = torch.broadcast_to(attention_mask, (batch_size, 1, seq_len, seq_len))
    visible = mask if mask.dtype == torch.bool else mask == 0
    qk = indexer.index_qk_proj(hidden_states)
    q_width = indexer.index_n_heads * indexer.index_head_dim
    q, raw_keys = torch.split(qk, [q_width, indexer.index_head_dim], dim=-1)
    q = indexer.q_layernorm(q.reshape(batch_size, seq_len, indexer.index_n_heads, indexer.index_head_dim))
    cos, sin = position_embeddings
    q = apply_rotary_pos_emb(q, cos=cos[:, -seq_len:], sin=sin[:, -seq_len:], unsqueeze_dim=2)
    selected = torch.full(
        (batch_size, seq_len, indexer.token_budget + indexer.compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    for batch_idx in range(batch_size):
        for query_idx in range(seq_len):
            visible_indices = visible[batch_idx, 0, query_idx].nonzero(as_tuple=True)[0]
            complete = visible_indices.numel() // indexer.compress_ratio
            block_tokens = visible_indices[: complete * indexer.compress_ratio].reshape(
                complete, indexer.compress_ratio
            )
            if complete:
                pooled = raw_keys[batch_idx, block_tokens].float().mean(dim=1).to(raw_keys.dtype)
                pooled = indexer.k_layernorm(pooled)
                starts = block_tokens[:, 0]
                keys = apply_rotary_pos_emb(
                    pooled.unsqueeze(1), cos=cos[batch_idx, starts], sin=sin[batch_idx, starts]
                ).squeeze(1)
                scores = torch.matmul(q[batch_idx, query_idx].float(), keys.float().T).T
                scores = torch.relu(scores).sum(-1) / indexer.index_head_dim**0.5
                top = scores.topk(min(indexer.block_topk, complete)).indices
                tokens = block_tokens[top].flatten()
            else:
                tokens = visible_indices[:0]
            tokens = torch.cat((tokens, visible_indices[complete * indexer.compress_ratio :]))
            selected[batch_idx, query_idx, : tokens.numel()] = tokens.to(torch.int32)
    return selected


def gather_qsa_selected_indices(local_indices: torch.Tensor, group, world_size: int) -> torch.Tensor:
    """Assemble local query selections into global query order."""
    if world_size == 1:
        return local_indices
    gathered = [torch.empty_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered, local_indices, group=group)
    return torch.cat(gathered, dim=1)


__all__ = ["compact_qsa_select", "explicit_mask_qsa_select", "gather_qsa_selected_indices"]
