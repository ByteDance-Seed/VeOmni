# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""Modeling-side MoE fallbacks that do not go through ``VeomniOp``.

EP dispatch stays in ``veomni.distributed.moe``. This module only runs the
non-SiLU expert loop, and calls that dispatch when weights are already sharded.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class _MergedExpertsActFnEP:
    """Local non-SiLU expert math on tokens already dispatched by EP."""

    @staticmethod
    def apply(
        permute_tokens: Tensor,
        cumsum: Tensor,
        gate_up_proj: Tensor,
        down_proj: Tensor,
        act_fn: Callable[[Tensor], Tensor],
        swiglu_limit: float | None = None,
    ) -> Tensor:
        """Run ``down(act_fn(gate) * up)`` on each local expert slice.

        Routing weights are applied later by ``tokens_post_all2all``.
        """
        num_local = gate_up_proj.shape[0]
        ends = cumsum.to(dtype=torch.long)
        starts = permute_tokens.new_zeros(num_local, dtype=torch.long)
        if num_local > 1:
            starts[1:] = ends[:-1]
        pieces: list[Tensor] = []
        for expert_idx in range(num_local):
            start = int(starts[expert_idx])
            end = int(ends[expert_idx])
            current_state = permute_tokens[start:end]
            gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = _gated_expert_activation(gate, up, act_fn, swiglu_limit)
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
            pieces.append(current_hidden_states.to(permute_tokens.dtype))
        output = torch.cat(pieces, dim=0)
        # Empty ranks still have to keep ``permute_tokens`` on the graph so the
        # EP all-to-all backward runs on every rank.
        return output + permute_tokens * 0


def _gated_expert_activation(
    gate: Tensor,
    up: Tensor,
    act_fn: Callable[[Tensor], Tensor],
    swiglu_limit: float | None,
) -> Tensor:
    """Apply optional DSV4-style clamp, then ``act_fn(gate) * up``."""
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return act_fn(gate) * up


def merged_experts_act_fn_forward(
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    act_fn: Callable[[Tensor], Tensor],
    num_experts: int,
    swiglu_limit: float | None = None,
) -> Tensor:
    """HF merged-expert loop: ``down(act_fn(gate) * up)`` then routing weights.

    Expert-parallel shards store only local rows on ``gate_up_proj``. Those
    tokens must go through the same EP dispatch as ``veomni_moe``; indexing
    with global expert ids would read past the local shard.
    Optional ``swiglu_limit`` applies DeepSeek-V4's gate/up clamp first.
    """
    local_experts = gate_up_proj.shape[0]
    if local_experts != num_experts:
        from veomni.distributed.moe import dispatch_to_ep_class
        from veomni.distributed.parallel_state import get_parallel_state

        state = get_parallel_state()
        if not state.ep_enabled or local_experts != num_experts // state.ep_size:
            raise ValueError(
                "non-SiLU MoE fallback received expert-parallel sharded weights "
                f"(local {local_experts} vs global {num_experts}). "
                "Enable EP so tokens are dispatched to local experts, or use silu."
            )
        return dispatch_to_ep_class(
            _MergedExpertsActFnEP,
            num_experts,
            top_k_weights,
            top_k_index,
            hidden_states,
            gate_up_proj,
            down_proj,
            act_fn,
            swiglu_limit,
        )

    final_hidden_states = hidden_states.new_zeros(hidden_states.shape)
    with torch.no_grad():
        expert_mask = nn.functional.one_hot(top_k_index, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = _gated_expert_activation(gate, up, act_fn, swiglu_limit)
        current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states
