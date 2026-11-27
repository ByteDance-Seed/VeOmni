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

"""Packed tensor preparation and block masks for padded MiniMax attention backends."""

from collections.abc import Callable

import torch


def _packed_to_padded_indices(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``[T]`` packed-to-flat-padded index map and the ``[B, Smax]`` padding mask (True=pad)."""
    if not isinstance(max_seqlen, int):
        raise TypeError(
            "MiniMax sparse attention max_seqlen must be a Python int to avoid a device-to-host sync, "
            f"got {type(max_seqlen).__name__}."
        )

    num_sequences = cu_seqlens.numel() - 1
    # NPU keeps FlashAttention metadata on CPU, whereas the scatter indices must
    # live next to the tensors they index. This is a device copy, never a host
    # scalar read.
    device_cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long, non_blocking=True)
    sequence_lengths = device_cu_seqlens.diff()

    token_indices = torch.arange(total_length, device=device)
    sequence_indices = torch.repeat_interleave(
        torch.arange(num_sequences, device=device),
        sequence_lengths,
        output_size=total_length,
    )
    sequence_positions = token_indices - device_cu_seqlens[sequence_indices]
    indices = sequence_indices * max_seqlen + sequence_positions

    padding = torch.arange(max_seqlen, device=device).unsqueeze(0) >= sequence_lengths.unsqueeze(1)
    return indices, padding


def _unpack(packed: torch.Tensor, indices: torch.Tensor, padding: torch.Tensor, fill: float = 0) -> torch.Tensor:
    """Scatter ``[T, ...]`` packed data into right-padded ``[B, Smax, ...]`` data."""
    batch_size, max_seqlen = padding.shape
    trailing_shape = packed.shape[1:]
    padded = packed.new_full((batch_size * max_seqlen, *trailing_shape), fill)
    padded.index_copy_(0, indices, packed)
    return padded.view(batch_size, max_seqlen, *trailing_shape)


def _pack(padded: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, Smax, ...]`` data back into its ``[T, ...]`` packed order."""
    return padded.reshape(-1, *padded.shape[2:]).index_select(0, indices)


def _expand_selected_blocks(
    selected: torch.Tensor,
    *,
    block_size: int,
    seq_len: int,
    via_bfloat16: bool,
) -> torch.Tensor:
    """Expand ``[B, G, S, num_key_blocks]`` block flags to a ``[B, G, S, S]`` token-level keep mask."""
    if via_bfloat16:
        # Boolean ``repeat_interleave`` along the key axis is very slow on NPU; 0/1 are
        # exact in bfloat16, so expanding there and casting back gives the same mask.
        return selected.to(torch.bfloat16).repeat_interleave(block_size, dim=-1)[..., :seq_len].bool()
    return selected.repeat_interleave(block_size, dim=-1)[..., :seq_len]


def _block_causal_keep(
    block_indices: torch.Tensor,
    padding: torch.Tensor,
    *,
    via_bfloat16: bool,
    block_size: int,
) -> torch.Tensor:
    """Expand ``[B, G, S, K]`` block indices into a ``[B, G, S, S]`` keep mask with padding and causality."""
    batch, num_groups, seq_len, _ = block_indices.shape
    num_key_blocks = -(-seq_len // block_size)

    # `-1` slots land in a throwaway column that is dropped afterwards.
    safe = block_indices.long().masked_fill(block_indices < 0, num_key_blocks)
    selected = torch.zeros(
        (batch, num_groups, seq_len, num_key_blocks + 1), dtype=torch.bool, device=block_indices.device
    )
    selected.scatter_(-1, safe, True)
    block_keep = _expand_selected_blocks(
        selected[..., :num_key_blocks],
        block_size=block_size,
        seq_len=seq_len,
        via_bfloat16=via_bfloat16,
    )

    positions = torch.arange(seq_len, device=block_indices.device)
    causal_keep = positions[None, :] <= positions[:, None]
    keep = block_keep & ~padding[:, None, None, :] & causal_keep[None, None]
    # Padded query rows select no block. Left fully masked, the dtype-min mask
    # overflows to -inf when the float32 softmax runs on float64 inputs, and the
    # resulting NaN rows leak into K/V gradients through a zero upstream grad.
    # Let them attend everywhere instead: `_pack` drops their outputs.
    return keep | padding[:, None, :, None]


def _run_padded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    attend: Callable,
    via_bfloat16: bool,
    block_size: int,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    num_query_heads, num_groups = q.shape[1], k.shape[1]
    if num_query_heads % num_groups != 0:
        raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_groups}).")
    if block_indices.shape[1] != num_groups:
        raise ValueError(
            f"MiniMax block_indices must have one selection per KV head ({num_groups}), got {block_indices.shape[1]}."
        )

    indices, padding = _packed_to_padded_indices(cu_seqlens, max_seqlen, q.shape[0], q.device)
    padded_q, padded_k, padded_v = (_unpack(tensor, indices, padding).transpose(1, 2) for tensor in (q, k, v))
    padded_block_indices = _unpack(block_indices, indices, padding, fill=-1).transpose(1, 2)
    keep = _block_causal_keep(
        padded_block_indices,
        padding,
        via_bfloat16=via_bfloat16,
        block_size=block_size,
    )

    output = attend(padded_q, padded_k, padded_v, keep, scale=scale, dropout_p=dropout_p)
    return _pack(output, indices)
