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

"""standard QSA dense-mask eager reference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    """Expand each KV head into ``repeats`` query-head rows."""
    if repeats == 1:
        return hidden_states
    batch_size, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None].expand(batch_size, kv_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch_size, kv_heads * repeats, seq_len, head_dim)


def wrapper(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_indices: Tensor,
    *,
    scaling: float,
    dropout: float = 0.0,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Expand the compact selection to a dense mask and run quadratic GQA.

    Q/K/V are ``[B, H, S, D]`` and *selected_indices* is ``[B, S, K]`` of
    global K/V token indices, where ``-1`` marks an unused slot. Query heads
    may be a multiple of KV heads. The output is ``[B, S, H, D]``.

    Like the native Transformers Qwen4-Exp eager path, this materializes the
    full ``[B, H, S, S]`` score tensor. It is a portable numerical reference,
    not a memory-efficient sparse backend. Rows that select nothing produce a
    zero output instead of a uniform average over masked slots.
    """
    kv_len = key.shape[2]
    query_heads = query.shape[1]
    selected_token_mask = torch.zeros(
        (*selected_indices.shape[:-1], kv_len + 1),
        dtype=torch.bool,
        device=selected_indices.device,
    )
    scatter_indices = torch.where(selected_indices >= 0, selected_indices, kv_len)
    allowed = selected_token_mask.scatter_(-1, scatter_indices.long(), True)[..., :kv_len][:, None]

    key_states = _repeat_kv(key, query_heads // key.shape[1])
    value_states = _repeat_kv(value, query_heads // value.shape[1])
    attention_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.is_floating_point():
            attention_weights = attention_weights + attention_mask
        else:
            allowed = allowed & attention_mask
    attention_weights = attention_weights.masked_fill(~allowed, torch.finfo(attention_weights.dtype).min)
    probabilities = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities.masked_fill(~allowed, 0)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    if dropout:
        probabilities = F.dropout(probabilities, p=dropout)
    return torch.matmul(probabilities, value_states).transpose(1, 2).contiguous(), attention_weights
