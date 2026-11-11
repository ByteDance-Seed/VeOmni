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

"""BAGEL two-route MoT async Ulysses QKV: routed project, QK norm, RoPE, then A2A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from ......distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)
from ......distributed.sequence_parallel.ulysses import all_to_all_tensor
from ......distributed.sequence_parallel.utils import (
    padding_tensor_for_seqeunce_parallel,
    unpadding_tensor_for_seqeunce_parallel,
)
from .....compound import InnerHandle, append_inner, resolve_inner_op, take_inner
from .....registry import OpEntry, SavedState
from ...shared.backward import linear_backward, reduce_repeated_kv_gradient


@dataclass
class _BagelQKVMeta:
    """Non-tensor metadata and nested-op specs for two-route MoT QKV backward."""

    sp_group: Any
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    repeat_kv: int
    has_q_bias_0: bool
    has_k_bias_0: bool
    has_v_bias_0: bool
    has_q_bias_1: bool
    has_k_bias_1: bool
    has_v_bias_1: bool
    rms: OpEntry
    rope: OpEntry
    rms_q_0: tuple[Any, int]
    rms_q_1: tuple[Any, int]
    rms_k_0: tuple[Any, int]
    rms_k_1: tuple[Any, int]
    rope_q: tuple[Any, int]
    rope_k: tuple[Any, int]


def _resolve_sp_group_and_size(group: Any) -> tuple[Any, int]:
    """Return the active Ulysses process group and its world size."""
    sp_group = get_ulysses_sequence_parallel_group() if group is None else group
    if sp_group is None:
        raise ValueError("Async MoT Ulysses requires an initialized Ulysses process group.")
    ulysses_size = (
        dist.get_world_size(sp_group)
        if isinstance(sp_group, dist.ProcessGroup)
        else get_ulysses_sequence_parallel_world_size()
    )
    return sp_group, ulysses_size


def _project_route(
    hidden_states: Tensor,
    indices: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    num_heads: int,
    head_dim: int,
) -> Tensor:
    """Project rows selected by ``indices`` and reshape to ``[tokens, heads, head_dim]``."""
    projected = F.linear(hidden_states.index_select(0, indices), weight, bias)
    return projected.reshape(-1, num_heads, head_dim)


def _rotate_single_forward(
    rope: OpEntry,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, SavedState]:
    """Apply full RoPE to one ``[tokens, heads, head_dim]`` tensor before all-to-all."""
    q_in = x.transpose(0, 1).unsqueeze(0)
    dummy_k = q_in.new_zeros((1, 1, x.shape[0], x.shape[2]))
    (q_out, _), saved = rope.forward(q_in, dummy_k, cos.unsqueeze(0), sin.unsqueeze(0), None, 1)
    return q_out.squeeze(0).transpose(0, 1).contiguous(), saved


def _rotate_single_backward(
    rope: OpEntry,
    grad_output: Tensor,
    saved: SavedState,
) -> Tensor:
    """Invert ``_rotate_single_forward`` on a ``[tokens, heads, head_dim]`` gradient."""
    grad_q = grad_output.transpose(0, 1).unsqueeze(0)
    dummy_grad_k = grad_q.new_zeros((1, 1, grad_output.shape[0], grad_output.shape[2]))
    dq, *_ = rope.backward((grad_q, dummy_grad_k), saved)
    return dq.squeeze(0).transpose(0, 1).contiguous()


def forward(
    hidden_states: Tensor,
    route_0_indices: Tensor,
    route_1_indices: Tensor,
    position_cos: Tensor,
    position_sin: Tensor,
    q_weight_0: Tensor,
    q_bias_0: Tensor | None,
    k_weight_0: Tensor,
    k_bias_0: Tensor | None,
    v_weight_0: Tensor,
    v_bias_0: Tensor | None,
    q_norm_weight_0: Tensor,
    k_norm_weight_0: Tensor,
    q_weight_1: Tensor,
    q_bias_1: Tensor | None,
    k_weight_1: Tensor,
    k_bias_1: Tensor | None,
    v_weight_1: Tensor,
    v_bias_1: Tensor | None,
    q_norm_weight_1: Tensor,
    k_norm_weight_1: Tensor,
    *,
    head_dim: int,
    rms_norm_eps: float,
    unpadded_dim_size: int,
    group: Any = None,
    rms_norm: InnerHandle = None,
    rope: InnerHandle = None,
) -> tuple[tuple[Tensor, Tensor, Tensor], SavedState]:
    """Two-route Q/K/V projection, per-route RMSNorm, RoPE, and async Ulysses exchange.

    Inputs are token-major ``[local_sequence, hidden_size]`` with local route
    index tensors ``route_0_indices`` and ``route_1_indices``. Outputs are
    ``[global_sequence, local_heads, head_dim]`` for Q, K, and V.
    """
    num_q_heads = q_weight_0.shape[0] // head_dim
    num_kv_heads = k_weight_0.shape[0] // head_dim

    sp_group, ulysses_size = _resolve_sp_group_and_size(group)
    if num_q_heads % ulysses_size != 0:
        raise ValueError(f"num_query_heads ({num_q_heads}) must be divisible by ulysses_size ({ulysses_size}).")
    if ulysses_size > num_kv_heads:
        if ulysses_size % num_kv_heads != 0:
            raise ValueError(
                f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({num_kv_heads})."
            )
        repeat_kv = ulysses_size // num_kv_heads
    else:
        if num_kv_heads % ulysses_size != 0:
            raise ValueError(
                f"num_key_value_heads ({num_kv_heads}) must be divisible by ulysses_size ({ulysses_size})."
            )
        repeat_kv = 1

    sequence_length = hidden_states.shape[0]
    rms_entry = resolve_inner_op(rms_norm, op="rms_norm", variant="standard")
    rope_entry = resolve_inner_op(rope, op="rope", variant="full")

    # Q projection + per-route RMSNorm + RoPE, followed by the first async
    # exchange. K/V projection executes while earlier communication is in flight.
    q_0 = _project_route(hidden_states, route_0_indices, q_weight_0, q_bias_0, num_q_heads, head_dim)
    q_1 = _project_route(hidden_states, route_1_indices, q_weight_1, q_bias_1, num_q_heads, head_dim)
    normalized_q_0, saved_rms_q_0 = rms_entry.forward(q_0, q_norm_weight_0, eps=rms_norm_eps)
    normalized_q_1, saved_rms_q_1 = rms_entry.forward(q_1, q_norm_weight_1, eps=rms_norm_eps)
    query = normalized_q_0.new_zeros((sequence_length, num_q_heads, head_dim))
    query.index_copy_(0, route_0_indices, normalized_q_0)
    query.index_copy_(0, route_1_indices, normalized_q_1)
    query, saved_rope_q = _rotate_single_forward(rope_entry, query, position_cos, position_sin)
    query_wait = all_to_all_tensor(
        query,
        scatter_dim=1,
        gather_dim=0,
        group=sp_group,
        async_op=True,
    )

    k_0 = _project_route(hidden_states, route_0_indices, k_weight_0, k_bias_0, num_kv_heads, head_dim)
    k_1 = _project_route(hidden_states, route_1_indices, k_weight_1, k_bias_1, num_kv_heads, head_dim)
    normalized_k_0, saved_rms_k_0 = rms_entry.forward(k_0, k_norm_weight_0, eps=rms_norm_eps)
    normalized_k_1, saved_rms_k_1 = rms_entry.forward(k_1, k_norm_weight_1, eps=rms_norm_eps)
    key = normalized_k_0.new_zeros((sequence_length, num_kv_heads, head_dim))
    key.index_copy_(0, route_0_indices, normalized_k_0)
    key.index_copy_(0, route_1_indices, normalized_k_1)
    key, saved_rope_k = _rotate_single_forward(rope_entry, key, position_cos, position_sin)
    if repeat_kv > 1:
        key = torch.repeat_interleave(key, dim=1, repeats=repeat_kv)
    key_wait = all_to_all_tensor(
        key,
        scatter_dim=1,
        gather_dim=0,
        group=sp_group,
        async_op=True,
    )

    v_0 = _project_route(hidden_states, route_0_indices, v_weight_0, v_bias_0, num_kv_heads, head_dim)
    v_1 = _project_route(hidden_states, route_1_indices, v_weight_1, v_bias_1, num_kv_heads, head_dim)
    value = v_0.new_zeros((sequence_length, num_kv_heads, head_dim))
    value.index_copy_(0, route_0_indices, v_0)
    value.index_copy_(0, route_1_indices, v_1)
    if repeat_kv > 1:
        value = torch.repeat_interleave(value, dim=1, repeats=repeat_kv)
    value_wait = all_to_all_tensor(
        value,
        scatter_dim=1,
        gather_dim=0,
        group=sp_group,
        async_op=True,
    )

    query = unpadding_tensor_for_seqeunce_parallel(query_wait(), 0, unpadded_dim_size, group=sp_group).contiguous()
    key = unpadding_tensor_for_seqeunce_parallel(key_wait(), 0, unpadded_dim_size, group=sp_group).contiguous()
    value = unpadding_tensor_for_seqeunce_parallel(value_wait(), 0, unpadded_dim_size, group=sp_group).contiguous()

    saved: list[Tensor | None] = [
        hidden_states,
        route_0_indices,
        route_1_indices,
        q_weight_0,
        k_weight_0,
        v_weight_0,
        q_weight_1,
        k_weight_1,
        v_weight_1,
    ]
    meta = _BagelQKVMeta(
        sp_group=sp_group,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        repeat_kv=repeat_kv,
        has_q_bias_0=q_bias_0 is not None,
        has_k_bias_0=k_bias_0 is not None,
        has_v_bias_0=v_bias_0 is not None,
        has_q_bias_1=q_bias_1 is not None,
        has_k_bias_1=k_bias_1 is not None,
        has_v_bias_1=v_bias_1 is not None,
        rms=rms_entry,
        rope=rope_entry,
        rms_q_0=append_inner(saved, saved_rms_q_0),
        rms_q_1=append_inner(saved, saved_rms_q_1),
        rms_k_0=append_inner(saved, saved_rms_k_0),
        rms_k_1=append_inner(saved, saved_rms_k_1),
        rope_q=append_inner(saved, saved_rope_q),
        rope_k=append_inner(saved, saved_rope_k),
    )
    return (query, key, value), SavedState(tuple(saved), meta)


def backward(grad_output: tuple[Tensor, Tensor, Tensor], saved: SavedState) -> tuple[Tensor | None, ...]:
    """Overlap inverse Ulysses all-to-all exchanges with routed RoPE, RMSNorm, and linear gradients."""
    meta = saved.metadata
    assert isinstance(meta, _BagelQKVMeta)
    (
        hidden_states,
        route_0_indices,
        route_1_indices,
        q_weight_0,
        k_weight_0,
        v_weight_0,
        q_weight_1,
        k_weight_1,
        v_weight_1,
        *rest_list,
    ) = saved.tensors
    rest = tuple(rest_list)
    saved_rms_q_0, rest = take_inner(rest, meta.rms_q_0)
    saved_rms_q_1, rest = take_inner(rest, meta.rms_q_1)
    saved_rms_k_0, rest = take_inner(rest, meta.rms_k_0)
    saved_rms_k_1, rest = take_inner(rest, meta.rms_k_1)
    saved_rope_q, rest = take_inner(rest, meta.rope_q)
    saved_rope_k, rest = take_inner(rest, meta.rope_k)

    grad_query, grad_key, grad_value = grad_output
    sp_group = meta.sp_group
    repeat_kv = meta.repeat_kv
    num_kv_heads = meta.num_kv_heads
    hidden_0 = hidden_states.index_select(0, route_0_indices)
    hidden_1 = hidden_states.index_select(0, route_1_indices)

    # V inverse exchange has no independent projection work before it, so
    # collect it first. K then overlaps V projection gradients, and Q
    # overlaps K normalization/projection gradients.
    grad_value = padding_tensor_for_seqeunce_parallel(grad_value.contiguous(), 0, group=sp_group)
    grad_value_wait = all_to_all_tensor(
        grad_value,
        scatter_dim=0,
        gather_dim=1,
        group=sp_group,
        async_op=True,
    )
    grad_value = grad_value_wait()
    if repeat_kv > 1:
        grad_value = reduce_repeated_kv_gradient(
            grad_value,
            num_kv_heads,
            repeat_kv,
            head_dimension=1,
        )

    grad_key = padding_tensor_for_seqeunce_parallel(grad_key.contiguous(), 0, group=sp_group)
    grad_key_wait = all_to_all_tensor(
        grad_key,
        scatter_dim=0,
        gather_dim=1,
        group=sp_group,
        async_op=True,
    )

    grad_hidden_v_0, grad_v_weight_0, grad_v_bias_0 = linear_backward(
        grad_value.index_select(0, route_0_indices),
        hidden_0,
        v_weight_0,
        has_bias=meta.has_v_bias_0,
    )
    grad_hidden_v_1, grad_v_weight_1, grad_v_bias_1 = linear_backward(
        grad_value.index_select(0, route_1_indices),
        hidden_1,
        v_weight_1,
        has_bias=meta.has_v_bias_1,
    )

    grad_key = grad_key_wait()
    if repeat_kv > 1:
        grad_key = reduce_repeated_kv_gradient(
            grad_key,
            num_kv_heads,
            repeat_kv,
            head_dimension=1,
        )

    grad_query = padding_tensor_for_seqeunce_parallel(grad_query.contiguous(), 0, group=sp_group)
    grad_query_wait = all_to_all_tensor(
        grad_query,
        scatter_dim=0,
        gather_dim=1,
        group=sp_group,
        async_op=True,
    )

    grad_key = _rotate_single_backward(meta.rope, grad_key, saved_rope_k)
    grad_k_0, grad_k_norm_weight_0 = meta.rms.backward(
        grad_key.index_select(0, route_0_indices),
        saved_rms_k_0,
    )
    grad_k_1, grad_k_norm_weight_1 = meta.rms.backward(
        grad_key.index_select(0, route_1_indices),
        saved_rms_k_1,
    )
    grad_hidden_k_0, grad_k_weight_0, grad_k_bias_0 = linear_backward(
        grad_k_0, hidden_0, k_weight_0, has_bias=meta.has_k_bias_0
    )
    grad_hidden_k_1, grad_k_weight_1, grad_k_bias_1 = linear_backward(
        grad_k_1, hidden_1, k_weight_1, has_bias=meta.has_k_bias_1
    )

    grad_query = grad_query_wait()
    grad_query = _rotate_single_backward(meta.rope, grad_query, saved_rope_q)
    grad_q_0, grad_q_norm_weight_0 = meta.rms.backward(
        grad_query.index_select(0, route_0_indices),
        saved_rms_q_0,
    )
    grad_q_1, grad_q_norm_weight_1 = meta.rms.backward(
        grad_query.index_select(0, route_1_indices),
        saved_rms_q_1,
    )
    grad_hidden_q_0, grad_q_weight_0, grad_q_bias_0 = linear_backward(
        grad_q_0, hidden_0, q_weight_0, has_bias=meta.has_q_bias_0
    )
    grad_hidden_q_1, grad_q_weight_1, grad_q_bias_1 = linear_backward(
        grad_q_1, hidden_1, q_weight_1, has_bias=meta.has_q_bias_1
    )

    grad_hidden_states = torch.zeros_like(hidden_states)
    grad_hidden_states.index_add_(
        0,
        route_0_indices,
        grad_hidden_q_0 + grad_hidden_k_0 + grad_hidden_v_0,
    )
    grad_hidden_states.index_add_(
        0,
        route_1_indices,
        grad_hidden_q_1 + grad_hidden_k_1 + grad_hidden_v_1,
    )

    return (
        grad_hidden_states,
        None,
        None,
        None,
        None,
        grad_q_weight_0,
        grad_q_bias_0,
        grad_k_weight_0,
        grad_k_bias_0,
        grad_v_weight_0,
        grad_v_bias_0,
        grad_q_norm_weight_0,
        grad_k_norm_weight_0,
        grad_q_weight_1,
        grad_q_bias_1,
        grad_k_weight_1,
        grad_k_bias_1,
        grad_v_weight_1,
        grad_v_bias_1,
        grad_q_norm_weight_1,
        grad_k_norm_weight_1,
    )
