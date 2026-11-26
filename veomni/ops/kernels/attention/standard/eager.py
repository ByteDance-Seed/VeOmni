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

"""Pure-PyTorch eager attention implementation with optional Ulysses SP support."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .....distributed.parallel_state import get_parallel_state
from ..helper import (
    extract_packed_varlen_cu_seqlens,
    packed_split_attention_forward,
)
from ..ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
    slice_ulysses_head_auxiliary,
)


def _repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Expand ``[B, H_kv, S, D]`` key/value tensors to match query head count."""
    if num_key_value_groups == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * num_key_value_groups, seq_len, head_dim)


def _slice_ulysses_head_tensor(
    tensor: torch.Tensor | None,
    *,
    query_head_count: int,
    local_query_head_count: int,
    group: dist.ProcessGroup,
    name: str,
) -> torch.Tensor | None:
    """Select this rank's query-head slice of a 3-D/4-D mask or bias after head scattering."""
    if tensor is None or tensor.ndim < 3:
        return tensor

    head_count = tensor.shape[-3]
    if head_count in (1, local_query_head_count):
        return tensor
    if head_count != query_head_count:
        raise ValueError(
            f"Eager attention with Ulysses requires the {name} head dimension to be 1, "
            f"the local query-head count ({local_query_head_count}), or the global query-head count "
            f"({query_head_count}); got {head_count}."
        )

    head_start = dist.get_rank(group) * local_query_head_count
    return tensor.narrow(-3, head_start, local_query_head_count)


def _build_unmasked_eager_mask(
    query_length: int,
    key_length: int,
    *,
    is_causal: bool,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a boolean visibility mask when no explicit ``attention_mask`` is supplied."""
    if not is_causal and sliding_window is None:
        return None
    if is_causal and query_length <= 1 and sliding_window is None:
        return None

    q_positions = torch.arange(query_length, device=device)[:, None] + (key_length - query_length)
    k_positions = torch.arange(key_length, device=device)[None, :]
    if is_causal:
        visible = q_positions >= k_positions
        if sliding_window is not None:
            visible = visible & ((q_positions - k_positions) < sliding_window)
    else:
        assert sliding_window is not None
        # Match ``sliding_window_bidirectional_overlay``: the boundary distance is visible.
        visible = (q_positions - k_positions).abs() <= sliding_window
    return visible[None, None]


def _fully_blocked_rows(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return rows whose keys are all invisible.

    Boolean masks use ``False`` for blocked keys. Additive masks use non-finite
    values, including ``-inf``. A row of finite minima is not fully blocked.
    """
    visible = attention_mask if attention_mask.dtype == torch.bool else torch.isfinite(attention_mask)
    return ~visible.any(dim=-1, keepdim=True)


def _softmax_without_fully_blocked_rows(
    attn_weights: torch.Tensor, fully_blocked: torch.Tensor | None
) -> torch.Tensor:
    """Softmax over keys, giving fully blocked rows probability 0 instead of NaN or a uniform distribution."""
    if fully_blocked is None:
        return F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    finite_logits = attn_weights.masked_fill(fully_blocked, 0)
    probabilities = F.softmax(finite_logits, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    return probabilities.masked_fill(fully_blocked, 0)


def _eager_attention_core(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pure-PyTorch scaled dot-product attention for one unpacked batch.

    Fully blocked rows receive zero probability. A boolean row with no ``True``
    would otherwise softmax to a uniform distribution, and an additive row of
    ``-inf`` would softmax to NaN.
    """
    if key.shape[1] != value.shape[1] or query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            "Eager attention requires query heads to be divisible by matching key/value heads; "
            f"got query={query.shape[1]}, key={key.shape[1]}, value={value.shape[1]}."
        )
    if key.shape[2] != value.shape[2]:
        raise ValueError(
            "Eager attention requires matching key and value sequence lengths; "
            f"got key={key.shape[2]}, value={value.shape[2]}."
        )

    num_key_value_groups = query.shape[1] // key.shape[1]
    key_states = _repeat_kv(key, num_key_value_groups)
    value_states = _repeat_kv(value, num_key_value_groups)

    scale = (query.shape[-1] ** -0.5) if scaling is None else scaling
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scale

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    position_bias = kwargs.get("position_bias")
    if position_bias is not None:
        if position_bias.ndim >= 2 and position_bias.shape[-1] > key_states.shape[-2]:
            position_bias = position_bias[..., : key_states.shape[-2]]
        attn_weights = attn_weights + position_bias

    fully_blocked = None
    if attention_mask is not None:
        if attention_mask.ndim >= 2 and attention_mask.shape[-1] > key_states.shape[-2]:
            attention_mask = attention_mask[..., : key_states.shape[-2]]
        fully_blocked = _fully_blocked_rows(attention_mask)
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, torch.finfo(attn_weights.dtype).min)
        else:
            attn_weights = attn_weights + attention_mask
    else:
        is_causal = kwargs.get("is_causal")
        if is_causal is None:
            is_causal = bool(getattr(module, "is_causal", False))
        visible = _build_unmasked_eager_mask(
            query.shape[-2],
            key_states.shape[-2],
            is_causal=bool(is_causal),
            sliding_window=sliding_window,
            device=query.device,
        )
        if visible is not None:
            fully_blocked = _fully_blocked_rows(visible)
            attn_weights = attn_weights.masked_fill(~visible, torch.finfo(attn_weights.dtype).min)

    training = bool(getattr(module, "training", False))
    s_aux = kwargs.get("s_aux")
    if s_aux is not None:
        if s_aux.ndim != 1 or s_aux.shape[0] != query.shape[1]:
            raise ValueError(
                "Eager attention requires `s_aux` to be a 1-D tensor matching query head count "
                f"({query.shape[1]}), got shape {tuple(s_aux.shape)}."
            )
        sinks = s_aux.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        attn_weights = F.dropout(scores, p=dropout, training=training).to(value_states.dtype)
    else:
        attn_weights = _softmax_without_fully_blocked_rows(attn_weights, fully_blocked).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run pure-PyTorch eager attention with optional Ulysses exchange and packed-segment splitting.

    Supports GQA expansion derived from Q/K head counts, explicit float
    additive or boolean masks, unmasked ``is_causal`` / ``sliding_window``
    visibility, ``softcap``, ``position_bias``, and attention sinks
    (``s_aux``). When Ulysses sequence parallelism is active and
    ``skip_ulysses`` is ``False``, Q/K/V are gathered along sequence and
    scattered across heads before attention (slicing global head-indexed
    masks, ``position_bias``, and ``s_aux`` to the local head shard), and the
    output is restored to sequence-sharded layout with ``(output, None)``.
    Unmasked batch-1 multi-segment ``cu_seqlens`` calls split Q/K/V per
    segment after any Ulysses gather and return ``(output, None)``.
    """
    kwargs, packed_cu_seqlens = extract_packed_varlen_cu_seqlens(
        kwargs,
        batch_size=query.shape[0],
        attention_mask=attention_mask,
    )

    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("Eager attention does not support query/key/value tensors with zero dimensions.")

    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses(skip_ulysses=skip_ulysses)
    if ulysses_enabled:
        query, key, value, query_head_count = prepare_ulysses_qkv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        local_query_head_count = query.shape[1]
        attention_mask = _slice_ulysses_head_tensor(
            attention_mask,
            query_head_count=query_head_count,
            local_query_head_count=local_query_head_count,
            group=parallel_state.ulysses_group,
            name="attention_mask",
        )
        position_bias = kwargs.get("position_bias")
        if position_bias is not None:
            kwargs = {
                **kwargs,
                "position_bias": _slice_ulysses_head_tensor(
                    position_bias,
                    query_head_count=query_head_count,
                    local_query_head_count=local_query_head_count,
                    group=parallel_state.ulysses_group,
                    name="position_bias",
                ),
            }
        s_aux = kwargs.get("s_aux")
        if s_aux is not None:
            kwargs = {
                **kwargs,
                "s_aux": slice_ulysses_head_auxiliary(
                    s_aux,
                    query_head_count=query_head_count,
                    local_query_head_count=local_query_head_count,
                    group=parallel_state.ulysses_group,
                ),
            }

    if packed_cu_seqlens is not None:
        cu_seqlens_q, cu_seqlens_k = packed_cu_seqlens
        attn_output, attn_weights = packed_split_attention_forward(
            _eager_attention_core,
            module,
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            **kwargs,
        )
    else:
        attn_output, attn_weights = _eager_attention_core(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            **kwargs,
        )

    if ulysses_enabled:
        attn_output = restore_ulysses_output(attn_output, group=parallel_state.ulysses_group)
        attn_weights = None

    return attn_output, attn_weights
