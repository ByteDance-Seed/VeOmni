"""Attention visibility metadata and FlexAttention masks for BAGEL Qwen2-MoT."""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask


_MOT_BLOCK_SIZE = 128


def build_mot_attention_metadata(
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Encode BAGEL packed visibility in three O(sequence) int32 rows.

    Row 0 identifies independent packed documents. Row 1 identifies spans
    whose tokens can attend bidirectionally within the span (``full`` and
    ``noise``). Row 2 identifies noise spans so their keys remain invisible
    outside that same noise span. ``-1`` means that a row does not apply.

    The model turns this compact representation into a native FlexAttention
    ``BlockMask`` without materializing an O(sequence²) token mask.
    """
    total_length = sum(sum(split_lens) for split_lens in sample_splits)
    metadata = torch.full((3, total_length), -1, device=device, dtype=torch.int32)
    cursor = 0
    span_id = 0
    for document_id, (split_lens, attn_modes) in enumerate(zip(sample_splits, sample_attn_modes, strict=True)):
        for length, mode in zip(split_lens, attn_modes, strict=True):
            if mode not in {"causal", "full", "noise"}:
                raise ValueError(f"Unsupported BAGEL attention mode: {mode!r}.")
            span_end = cursor + length
            metadata[0, cursor:span_end] = document_id
            if mode != "causal":
                metadata[1, cursor:span_end] = span_id
            if mode == "noise":
                metadata[2, cursor:span_end] = span_id
            cursor = span_end
            span_id += 1
    return metadata.contiguous()


def _validate_mot_attention_metadata(packed_attention_metadata: torch.Tensor) -> None:
    if packed_attention_metadata.ndim != 2 or packed_attention_metadata.shape[0] != 3:
        raise ValueError(
            "BAGEL Qwen2-MoT attention metadata must have shape [3, sequence], "
            f"got {tuple(packed_attention_metadata.shape)}."
        )
    if packed_attention_metadata.dtype != torch.int32:
        raise ValueError(
            f"BAGEL Qwen2-MoT attention metadata must use torch.int32, got {packed_attention_metadata.dtype}."
        )


def pad_mot_attention_metadata(
    packed_attention_metadata: torch.Tensor,
    padded_length: int,
) -> torch.Tensor:
    """Pad metadata with one isolated full-attention span for SP alignment.

    Giving padding its own document prevents real tokens from attending to it;
    leaving its noise ID at ``-1`` makes the padding block safe to classify as
    a regular clean block.
    """
    _validate_mot_attention_metadata(packed_attention_metadata)

    sequence_length = int(packed_attention_metadata.shape[1])
    if padded_length < sequence_length:
        raise ValueError(
            "BAGEL Qwen2-MoT padded attention length cannot shrink the sequence: "
            f"sequence_length={sequence_length}, padded_length={padded_length}."
        )
    if padded_length == sequence_length:
        return packed_attention_metadata

    padded = torch.full(
        (3, padded_length),
        -1,
        device=packed_attention_metadata.device,
        dtype=packed_attention_metadata.dtype,
    )
    padded[:, :sequence_length] = packed_attention_metadata
    padding_id = sequence_length
    padded[0, sequence_length:] = padding_id
    padded[1, sequence_length:] = padding_id
    return padded.contiguous()


def _metadata_block_ranges(
    token_ids: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize monotonic token IDs for conservative block classification."""
    padding = (-int(token_ids.numel())) % block_size
    if padding:
        token_ids = torch.nn.functional.pad(token_ids, (0, padding), value=-1)
    token_ids = token_ids.view(-1, block_size)
    valid = token_ids >= 0
    minimum = torch.where(valid, token_ids, torch.iinfo(token_ids.dtype).max).amin(dim=-1)
    maximum = token_ids.amax(dim=-1)
    return minimum, maximum, valid.any(dim=-1), valid.all(dim=-1)


def _ordered_block_indices(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense block-level predicate to FlexAttention ordered rows."""
    block_count = int(blocks.shape[-1])
    block_ids = torch.arange(block_count, device=blocks.device, dtype=torch.int32).expand(block_count, -1)
    sentinel = torch.full_like(block_ids, block_count)
    indices = torch.where(blocks, block_ids, sentinel).sort(dim=-1).values
    num_blocks = blocks.sum(dim=-1, dtype=torch.int32)
    return num_blocks.unsqueeze(0).unsqueeze(0), indices.unsqueeze(0).unsqueeze(0)


def build_mot_block_mask(packed_attention_metadata: torch.Tensor) -> BlockMask:
    """Build BAGEL's head-independent BlockMask from compact metadata.

    Token visibility is:
    ``same_document & (causal | same_full_span) & ~foreign_noise_key``.
    """
    _validate_mot_attention_metadata(packed_attention_metadata)

    sequence_length = int(packed_attention_metadata.shape[1])
    if sequence_length == 0:
        raise ValueError("BAGEL Qwen2-MoT attention metadata must contain at least one token.")
    document_ids, full_span_ids, noise_span_ids = packed_attention_metadata

    def mask_mod(batch_idx, head_idx, query_idx, key_idx):
        # MoT visibility is identical across batch and attention heads, but
        # FlexAttention requires the four-argument callback signature.
        del batch_idx, head_idx
        same_document = document_ids[query_idx] == document_ids[key_idx]
        causal = query_idx >= key_idx
        same_full_or_noise_span = (full_span_ids[query_idx] >= 0) & (
            full_span_ids[query_idx] == full_span_ids[key_idx]
        )
        foreign_noise_key = (noise_span_ids[key_idx] >= 0) & (noise_span_ids[query_idx] != noise_span_ids[key_idx])
        return same_document & (causal | same_full_or_noise_span) & ~foreign_noise_key

    doc_min, doc_max, doc_valid, doc_complete = _metadata_block_ranges(
        document_ids,
        block_size=_MOT_BLOCK_SIZE,
    )
    full_min, full_max, full_valid, full_complete = _metadata_block_ranges(
        full_span_ids,
        block_size=_MOT_BLOCK_SIZE,
    )
    _, noise_max, _, _ = _metadata_block_ranges(
        noise_span_ids,
        block_size=_MOT_BLOCK_SIZE,
    )

    # Range overlap produces a conservative candidate set: mixed boundary
    # blocks may be included here and are filtered token-by-token by mask_mod.
    same_document_block = (
        doc_valid[:, None]
        & doc_valid[None, :]
        & (doc_min[:, None] <= doc_max[None, :])
        & (doc_min[None, :] <= doc_max[:, None])
    )
    shared_full_span_block = (
        full_valid[:, None]
        & full_valid[None, :]
        & (full_min[:, None] <= full_max[None, :])
        & (full_min[None, :] <= full_max[:, None])
    )
    block_count = int(doc_min.numel())
    block_ids = torch.arange(block_count, device=packed_attention_metadata.device)
    causal_block = block_ids[:, None] >= block_ids[None, :]
    candidate_blocks = same_document_block & (causal_block | shared_full_span_block)

    # Full blocks bypass mask_mod. Mark a block pair full only when every token
    # is provably in one document and visibility is unconditional: either the
    # key block is strictly earlier and clean, or both blocks share one
    # bidirectional span.
    single_document_block = doc_complete & (doc_min == doc_max)
    same_single_document = (
        single_document_block[:, None] & single_document_block[None, :] & (doc_min[:, None] == doc_min[None, :])
    )
    single_full_span_block = full_complete & full_valid & (full_min == full_max)
    same_single_full_span = (
        single_full_span_block[:, None] & single_full_span_block[None, :] & (full_min[:, None] == full_min[None, :])
    )
    strictly_earlier_clean_key_block = (block_ids[:, None] > block_ids[None, :]) & (noise_max[None, :] < 0)
    full_blocks = candidate_blocks & same_single_document & (strictly_earlier_clean_key_block | same_single_full_span)
    partial_blocks = candidate_blocks & ~full_blocks

    partial_num_blocks, partial_indices = _ordered_block_indices(partial_blocks)
    full_num_blocks, full_indices = _ordered_block_indices(full_blocks)
    return BlockMask.from_kv_blocks(
        partial_num_blocks,
        partial_indices,
        full_num_blocks,
        full_indices,
        BLOCK_SIZE=_MOT_BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(sequence_length, sequence_length),
    )


__all__ = [
    "build_mot_attention_metadata",
    "build_mot_block_mask",
    "pad_mot_attention_metadata",
]
