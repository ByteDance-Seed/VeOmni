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

"""Eager reference implementation of the MiniMax M3 sparse attention operators.

The math is the Transformers ``MiniMaxM3VLIndexer`` / ``build_block_mask`` /
``eager_attention_forward`` reference, run on a temporary right-padded
``[B, Smax, ...]`` view of the packed input. The unpack/pack round trip exists
only because the reference math is batched; it never leaves this module.

On Ascend NPU tensors the masked attention runs through
``torch_npu.npu_fusion_attention`` with the same boolean keep mask; every other
device uses the PyTorch reference.
"""

import torch
import torch.nn.functional as F


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


def _expand_selected_blocks(
    selected: torch.Tensor,
    *,
    block_size: int,
    seq_len: int,
    heads_per_group: int,
    via_bfloat16: bool,
) -> torch.Tensor:
    """Expand ``[B, G, S, num_key_blocks]`` block flags to a ``[B, Hq, S, S]`` token-level keep mask."""
    if via_bfloat16:
        # Boolean ``repeat_interleave`` along the key axis is very slow on NPU; 0/1 are
        # exact in bfloat16, so expanding there and casting back gives the same mask.
        block_keep = selected.to(torch.bfloat16).repeat_interleave(block_size, dim=-1)[..., :seq_len].bool()
    else:
        block_keep = selected.repeat_interleave(block_size, dim=-1)[..., :seq_len]
    return block_keep.repeat_interleave(heads_per_group, dim=1)


def _block_causal_keep(
    block_indices: torch.Tensor,
    padding: torch.Tensor,
    *,
    num_query_heads: int,
    block_size: int,
) -> torch.Tensor:
    """Expand ``[B, G, S, K]`` block indices into a ``[B, Hq, S, S]`` keep mask with padding and causality."""
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
        heads_per_group=num_query_heads // num_groups,
        via_bfloat16=block_indices.device.type == "npu",
    )

    positions = torch.arange(seq_len, device=block_indices.device)
    causal_keep = positions[None, :] <= positions[:, None]
    keep = block_keep & ~padding[:, None, None, :] & causal_keep[None, None]
    # Padded query rows select no block. Left fully masked, the dtype-min mask
    # overflows to -inf when the float32 softmax runs on float64 inputs, and the
    # resulting NaN rows leak into K/V gradients through a zero upstream grad.
    # Let them attend everywhere instead: `_pack` drops their outputs.
    return keep | padding[:, None, :, None]


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep: torch.Tensor,
    *,
    scale: float,
    dropout_p: float,
) -> torch.Tensor:
    """Masked attention on ``[B, H, S, D]`` with local GQA; returns ``[B, S, Hq, D]``.

    NPU tensors use ``npu_fusion_attention`` (native GQA, ``True`` in ``atten_mask``
    means masked); other devices run the upstream eager math.
    """
    if query.device.type == "npu":
        import torch_npu

        output = torch_npu.npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            head_num=query.shape[1],
            input_layout="BNSD",
            atten_mask=(~keep).contiguous(),
            sparse_mode=1,
            scale=scale,
            keep_prob=1.0 - dropout_p,
        )[0]
        return output.transpose(1, 2).contiguous()

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
    block_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    block_size: int | None,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Causal (block-sparse when ``block_indices`` is given) attention on TND tensors; see the package docstring."""
    num_query_heads, num_groups = q.shape[1], k.shape[1]
    if num_query_heads % num_groups != 0:
        raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_groups}).")
    if block_indices is not None and block_indices.shape[1] != num_groups:
        raise ValueError(
            f"MiniMax block_indices must have one selection per KV head ({num_groups}), got {block_indices.shape[1]}."
        )

    indices, padding = _packed_to_padded_indices(cu_seqlens, max_seqlen, q.shape[0], q.device)
    padded_q, padded_k, padded_v = (_unpack(tensor, indices, padding).transpose(1, 2) for tensor in (q, k, v))

    if block_indices is None:
        positions = torch.arange(max_seqlen, device=q.device)
        causal_keep = positions[None, :] <= positions[:, None]
        keep = causal_keep[None, None] & ~padding[:, None, None, :]
    else:
        padded_block_indices = _unpack(block_indices, indices, padding, fill=-1).transpose(1, 2)
        keep = _block_causal_keep(
            padded_block_indices,
            padding,
            num_query_heads=num_query_heads,
            block_size=block_size,
        )

    output = _dense_attention(padded_q, padded_k, padded_v, keep, scale=scale, dropout_p=dropout_p)
    return _pack(output, indices)
