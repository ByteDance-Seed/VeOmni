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

"""Packed-sequence helpers for DeepSeek V4 compressed attention."""

from typing import Callable, NamedTuple

import torch


def build_packed_compression_metadata(
    reference: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_slices: tuple[tuple[int, int], ...],
    compress_rates: tuple[int, ...],
    block_bias_rates: tuple[int, ...] = (),
) -> dict[int, dict[str, torch.Tensor]]:
    """Build reusable packed window indices, ranges, and masks once per forward."""
    metadata = {}
    for compress_rate in dict.fromkeys(compress_rates):
        window_starts_list = [
            window_start
            for start, end in sequence_slices
            for window_start in range(start, end - compress_rate + 1, compress_rate)
        ]
        window_starts = torch.tensor(window_starts_list, device=reference.device, dtype=torch.long)
        offsets = torch.arange(compress_rate, device=reference.device, dtype=torch.long)
        window_indices = window_starts[:, None] + offsets[None, :]

        range_starts = torch.empty(position_ids.shape[1], device=reference.device, dtype=torch.int32)
        range_ends = torch.empty_like(range_starts)
        compressed_offset = 0
        for start, end in sequence_slices:
            compressed_count = (end - start) // compress_rate
            range_starts[start:end] = compressed_offset
            visible_counts = torch.minimum(
                (position_ids[0, start:end] + 1) // compress_rate,
                position_ids.new_tensor(compressed_count),
            )
            range_ends[start:end] = compressed_offset + visible_counts.to(torch.int32)
            compressed_offset += compressed_count

        rate_metadata = {
            "window_starts": window_starts,
            "window_indices": window_indices,
            "range_starts": range_starts,
            "range_ends": range_ends,
        }
        if compress_rate in block_bias_rates:
            entry_indices = torch.arange(compressed_offset, device=reference.device)
            allowed = (entry_indices[None, :] >= range_starts[:, None]) & (
                entry_indices[None, :] < range_ends[:, None]
            )
            block_bias = reference.new_full((1, 1, position_ids.shape[1], compressed_offset), float("-inf"))
            rate_metadata["block_bias"] = block_bias.masked_fill(allowed[None, None, :, :], 0.0)
        metadata[compress_rate] = rate_metadata
    return metadata


def compress_packed_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    head_dim: int,
    compress_rate: int,
    kv_norm: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    rope_layer_type: str,
    position_ids: torch.Tensor,
    packed_metadata: dict[str, torch.Tensor],
    *,
    overlap: bool,
    apply_rope: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Compress complete windows without crossing packed sequence boundaries.

    Dynamic batching represents a packed microbatch as ``[1, total_tokens]``
    and resets ``position_ids`` to zero at each sequence boundary. Selecting
    window ends from local positions keeps incomplete tails out of the next
    sequence and lets every operation stay on device.

    ``apply_rope`` is injected so callers in the generated modeling pass their
    module-global ``apply_rotary_pos_emb``, which ``device_patch.py`` may have
    swapped for the fused Triton backend. It is required rather than defaulted
    to the eager reference: a defaulted call site would silently keep eager
    while every other one is fused, which no test would catch.
    """
    if kv.shape[0] != 1:
        raise ValueError(f"Packed DeepSeek V4 compression expects batch size 1, got {kv.shape[0]}")

    window_starts = packed_metadata["window_starts"]
    if window_starts.numel() == 0:
        return kv.new_zeros((1, 0, head_dim))

    current_indices = packed_metadata["window_indices"]
    current_kv = kv[0, current_indices]
    current_gate = gate[0, current_indices] + position_bias.to(gate.dtype)

    if overlap:
        previous_indices = (current_indices - compress_rate).clamp_min(0)
        previous_kv = kv[0, previous_indices, :head_dim]
        previous_gate = gate[0, previous_indices, :head_dim] + position_bias[:, :head_dim].to(gate.dtype)
        has_previous_window = position_ids[0, window_starts] >= compress_rate
        previous_gate = previous_gate.masked_fill(~has_previous_window[:, None, None], float("-inf"))
        window_kv = torch.cat([previous_kv, current_kv[..., head_dim:]], dim=1)
        window_gate = torch.cat([previous_gate, current_gate[..., head_dim:]], dim=1)
    else:
        window_kv = current_kv
        window_gate = current_gate

    # `sum` follows autocast's fp32_set_opt_dtype policy, so an implicit `dtype`
    # silently returns fp32 under autocast and leaks out through `kv_norm` into
    # the bf16-only TileLang kernels. Accumulate in fp32 explicitly, cast back.
    compressed = kv_norm(
        (window_kv * window_gate.softmax(dim=1, dtype=torch.float32).to(window_kv.dtype))
        .sum(dim=1, dtype=torch.float32)
        .to(window_kv.dtype)
    ).unsqueeze(0)
    window_positions = position_ids[0, window_starts]
    cos, sin = rotary_emb(
        compressed,
        position_ids=window_positions.unsqueeze(0),
        layer_type=rope_layer_type,
    )
    return apply_rope(compressed.unsqueeze(1), cos, sin).squeeze(1)


def packed_compressed_causal_ranges(
    packed_metadata: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local compressed-KV interval visible to every packed query."""
    return packed_metadata["range_starts"], packed_metadata["range_ends"]


def packed_compressed_block_bias(
    packed_metadata: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build a block-diagonal causal mask for packed compressed entries."""
    return packed_metadata["block_bias"]


def scatter_topk_block_bias(
    compressed_kv: torch.Tensor,
    top_k_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Re-encode an indexer selection as an additive ``[B, 1, S, C]`` block bias."""
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0
    safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
    block_bias = compressed_kv.new_full((batch_size, 1, seq_len, compressed_len + 1), float("-inf"))
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return block_bias[..., :compressed_len]


def build_sparse_attention_indices(
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int,
    compressed_len: int,
    compressed_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Build compact sliding-window and compressed-KV candidates."""
    sliding_width = min(sliding_window, seq_len)
    query_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
    window_offsets = torch.arange(sliding_width - 1, -1, -1, device=device, dtype=torch.int32)
    sliding_indices = query_indices[:, None] - window_offsets[None, :]
    sliding_indices = sliding_indices.masked_fill(sliding_indices < 0, -1)
    sliding_indices = sliding_indices.unsqueeze(0).expand(batch_size, -1, -1)

    if compressed_len == 0:
        return sliding_indices.contiguous()

    if compressed_indices is None:
        compressed_indices = torch.arange(compressed_len, device=device, dtype=torch.int32)
        compressed_indices = compressed_indices.view(1, 1, -1).expand(batch_size, seq_len, -1)
    else:
        if compressed_indices.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "Compressed sparse indices must match the attention batch and sequence dimensions; "
                f"got {tuple(compressed_indices.shape)} for batch={batch_size}, seq_len={seq_len}"
            )
        compressed_indices = compressed_indices.to(device=device, dtype=torch.int32)

    compressed_indices = torch.where(
        compressed_indices >= 0,
        compressed_indices + seq_len,
        torch.full_like(compressed_indices, -1),
    )
    return torch.cat((sliding_indices, compressed_indices), dim=-1).contiguous()


class CompressedCandidates(NamedTuple):
    """How a compressor exposes its visible compressed entries to the index builder.

    Indexer-driven compressors select an explicit per-query set and already mark
    misses with ``-1``; the heavily-compressed path instead stays a contiguous
    ``[range_start, range_end)`` causal interval. At most one of the two is set.
    """

    topk_indices: torch.Tensor | None = None
    range_starts: torch.Tensor | None = None
    range_ends: torch.Tensor | None = None


def _broadcast_query_ranges(bound: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Normalize a per-query range bound to ``[batch, seq_len]``."""
    if bound.ndim == 1:
        bound = bound.unsqueeze(0)
    if bound.shape[-1] != seq_len:
        raise ValueError(f"Compressed range bounds must cover {seq_len} queries; got {tuple(bound.shape)}")
    return bound.expand(batch_size, -1).to(torch.int32)


def build_packed_sparse_attention_indices(
    *,
    position_ids: torch.Tensor,
    sliding_window: int,
    compressed_len: int,
    candidates: CompressedCandidates | None,
) -> torch.Tensor:
    """Build fully validated sparse candidates without materializing a dense mask.

    The dense path derives validity by scanning an ``[B, 1, S, S + C]`` additive
    mask, which costs O(S^2) memory and bandwidth on every layer. Every constraint
    that mask encodes is already available in closed form:

    * sliding causality and packed-sample isolation collapse into the single test
      ``offset <= position_ids[q]``. ``position_ids`` restarts at zero on each
      packed sample, so ``q - position_ids[q]`` is that sample's first token and
      any larger offset would reach into the previous sample.
    * compressed entries are either an explicit indexer selection that already
      marks misses with ``-1``, or a contiguous ``[range_start, range_end)``
      interval computed from the same metadata that built the block bias.

    Callers must guarantee the padding mask is all ones over the packed length,
    which the FlashAttention varlen collator does by construction -- boundaries
    travel through ``position_ids`` and ``cu_seq_lens`` instead.
    """
    if position_ids.ndim != 2:
        raise ValueError(f"Packed sparse indices need [batch, seq_len] position ids; got {position_ids.ndim} dims")

    batch_size, seq_len = position_ids.shape
    device = position_ids.device
    sliding_width = min(sliding_window, seq_len)
    positions = position_ids.to(torch.int32)
    query_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
    window_offsets = torch.arange(sliding_width - 1, -1, -1, device=device, dtype=torch.int32)

    sliding_indices = (query_indices[None, :, None] - window_offsets[None, None, :]).expand(batch_size, -1, -1)
    sliding_indices = sliding_indices.masked_fill(window_offsets[None, None, :] > positions[:, :, None], -1)

    if compressed_len == 0 or candidates is None:
        return sliding_indices.contiguous()

    if candidates.topk_indices is not None:
        compressed_indices = candidates.topk_indices.to(device=device, dtype=torch.int32)
        if compressed_indices.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "Compressed sparse indices must match the attention batch and sequence dimensions; "
                f"got {tuple(compressed_indices.shape)} for batch={batch_size}, seq_len={seq_len}"
            )
        compressed_indices = torch.where(compressed_indices >= 0, compressed_indices + seq_len, -1)
    elif candidates.range_starts is not None and candidates.range_ends is not None:
        starts = _broadcast_query_ranges(candidates.range_starts, batch_size, seq_len)
        ends = _broadcast_query_ranges(candidates.range_ends, batch_size, seq_len)
        entry_indices = torch.arange(compressed_len, device=device, dtype=torch.int32)
        visible = (entry_indices[None, None, :] >= starts[:, :, None]) & (
            entry_indices[None, None, :] < ends[:, :, None]
        )
        compressed_indices = torch.where(visible, entry_indices[None, None, :] + seq_len, -1)
    else:
        return sliding_indices.contiguous()

    return torch.cat((sliding_indices, compressed_indices), dim=-1).contiguous()


def mask_sparse_attention_indices(
    attention_mask: torch.Tensor,
    sparse_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply the canonical attention mask to compact sparse candidates."""
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


def isolate_packed_causal_mask_(
    causal_mask: torch.Tensor | None,
    sequence_slices: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Mask previous packed samples in an explicit causal/sliding-window mask."""
    if causal_mask is None or not isinstance(causal_mask, torch.Tensor):
        raise ValueError("DeepSeek V4 packed training requires an explicit tensor attention mask")
    blocked_value = False if causal_mask.dtype == torch.bool else torch.finfo(causal_mask.dtype).min
    for start, end in sequence_slices[1:]:
        causal_mask[..., start:end, :start] = blocked_value
    return causal_mask


__all__ = [
    "CompressedCandidates",
    "build_packed_compression_metadata",
    "build_packed_sparse_attention_indices",
    "build_sparse_attention_indices",
    "compress_packed_windows",
    "isolate_packed_causal_mask_",
    "mask_sparse_attention_indices",
    "packed_compressed_block_bias",
    "packed_compressed_causal_ranges",
    "scatter_topk_block_bias",
]
