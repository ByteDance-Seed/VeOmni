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

"""Correctness-oriented reference attention primitives.

Not a production fallback -- these implementations trade throughput for the
smallest, easiest-to-audit path and exist so kernel/fused-attention outputs
can be diffed against a mathematically-defined ground truth.
"""

import torch
from torch import nn


def dense_gca_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dense generalized causal attention from a boolean allowed-edge mask.

    Handles grouped-query attention by expanding the K/V heads to match the Q
    heads via ``module.num_key_value_groups`` (the standard HF attention module
    attribute) and materialises the full ``[batch, heads, q_len, kv_len]``
    score matrix -- suitable for reference / parity testing, not for
    long-sequence training.

    The ``attention_mask`` is a boolean edge mask (``True`` = attention
    allowed) shaped ``[batch, q_len, kv_len]`` or ``[batch, 1, q_len,
    kv_len]``; disallowed edges get the dtype's minimum before softmax so the
    normalisation is exact.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, sequence, head_dim].")
    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes.")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query and key/value batch, sequence, and head dimensions must match.")
    if attention_mask.dtype != torch.bool:
        raise TypeError("The dense GCA reference mask must be boolean.")
    if attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(1)
    expected_mask_shape = (query.shape[0], 1, query.shape[2], key.shape[2])
    if attention_mask.shape != expected_mask_shape:
        raise ValueError(f"The dense GCA reference mask must have shape {expected_mask_shape}.")

    key = _repeat_kv(key, module.num_key_value_groups)
    value = _repeat_kv(value, module.num_key_value_groups)
    attention_scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    attention_scores = attention_scores.masked_fill(
        ~attention_mask,
        torch.finfo(attention_scores.dtype).min,
    )
    attention_weights = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_output = torch.matmul(attention_weights, value)
    return attention_output.transpose(1, 2).contiguous(), attention_weights


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch_size, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(
        batch_size,
        num_key_value_heads,
        repeats,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * repeats, sequence_length, head_dim)


__all__ = ["dense_gca_attention_forward"]
