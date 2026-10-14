"""Shared projection helpers for Qwen4-Exp QSA kernels."""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_query_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rope, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated_half = _rotate_half(rope)
    rotated = rope * cos.unsqueeze(2) + rotated_half * sin.unsqueeze(2)
    return torch.cat((rotated, passthrough), dim=-1)


def project_qsa_indexer_inputs(
    indexer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project and normalize the inputs consumed by the QSA indexer."""
    batch_size, sequence_length, _ = hidden_states.shape
    full_cos, full_sin = position_embeddings
    current_cos = full_cos[:, -sequence_length:, :]
    current_sin = full_sin[:, -sequence_length:, :]

    projected = indexer.index_qk_proj(hidden_states)
    query_width = indexer.index_n_heads * indexer.index_head_dim
    query, raw_keys = projected.split(
        [query_width, indexer.index_kv_heads * indexer.index_head_dim],
        dim=-1,
    )
    query = query.view(batch_size, sequence_length, indexer.index_n_heads, indexer.index_head_dim)
    raw_keys = raw_keys.view(batch_size, sequence_length, indexer.index_kv_heads, indexer.index_head_dim)
    if indexer.index_kv_heads != 1:
        raise ValueError("Qwen4-Exp QSA kernels require indexer_kv_heads == 1")
    raw_keys = raw_keys.squeeze(2)
    query = _apply_query_rope(indexer.q_layernorm(query), current_cos, current_sin)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, indexer.layer_idx)
    return query, raw_keys, full_cos, full_sin
