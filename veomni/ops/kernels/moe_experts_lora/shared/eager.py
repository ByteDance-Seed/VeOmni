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
# See the License for the specific language governing permissions and
# limitations under the License.

"""shared MoE-LoRA eager math. Regular autograd, no custom backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..routing import group_routing_assignments


def _shared_lora_mid(
    state: Tensor,
    fc1_weight: Tensor,
    gate_delta: Tensor,
    up_delta: Tensor,
) -> Tensor:
    """``act(gate) * up`` for one expert, with the shared LoRA deltas added first."""
    gate_up = F.linear(state, fc1_weight) + torch.cat([gate_delta, up_delta], dim=-1)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


class _SharedLoRAExpertMathEP:
    """Shared-LoRA expert math on tokens already dispatched by EP.

    Operates on ``permute_tokens``: each row is one (token, slot) pair routed to a
    *local* expert, and ``cumsum`` bounds the rows per local expert. Routing weights
    are applied afterwards by ``tokens_post_all2all``, so this loop does not multiply
    by them -- which is also why the base and the down-LoRA delta stay linear in
    ``mid`` and the two conventions agree.

    The LoRA pairs are 2-D and shared across experts, so they are replicated on every
    rank (unlike the 3-D per-expert adapters of ``independent`` mode). Their gradients
    therefore need a cross-rank sum, which the wrapper installs a post-accumulate hook
    for; this loop only has to produce the local partials.
    """

    @staticmethod
    def apply(
        permute_tokens: Tensor,
        cumsum: Tensor,
        fc1_1_2_weight: Tensor,
        fc2_weight: Tensor,
        lora_a_gate: Tensor,
        lora_b_gate: Tensor,
        lora_a_up: Tensor,
        lora_b_up: Tensor,
        lora_a_down: Tensor,
        lora_b_down: Tensor,
        lora_scale_gate: float,
        lora_scale_up: float,
        lora_scale_down: float,
    ) -> Tensor:
        """Run ``down(act(gate) * up)`` plus the shared LoRA deltas on local experts.

        Routing weights are applied later by ``tokens_post_all2all``.
        """
        scale_gate = permute_tokens.new_tensor(lora_scale_gate)
        scale_up = permute_tokens.new_tensor(lora_scale_up)
        scale_down = permute_tokens.new_tensor(lora_scale_down)

        # Both deltas read every dispatched row, so hoist them out of the expert loop.
        gate_delta = F.linear(F.linear(permute_tokens, lora_a_gate), lora_b_gate) * scale_gate
        up_delta = F.linear(F.linear(permute_tokens, lora_a_up), lora_b_up) * scale_up

        num_local = fc1_1_2_weight.shape[0]
        ends = cumsum.to(dtype=torch.long)
        starts = permute_tokens.new_zeros(num_local, dtype=torch.long)
        if num_local > 1:
            starts[1:] = ends[:-1]
        pieces: list[Tensor] = []
        for expert_idx in range(num_local):
            start = int(starts[expert_idx])
            end = int(ends[expert_idx])
            mid = _shared_lora_mid(
                permute_tokens[start:end],
                fc1_1_2_weight[expert_idx],
                gate_delta[start:end],
                up_delta[start:end],
            )
            lora_delta_down = F.linear(F.linear(mid, lora_a_down), lora_b_down) * scale_down
            current = F.linear(mid, fc2_weight[expert_idx]) + lora_delta_down
            pieces.append(current.to(permute_tokens.dtype))
        output = torch.cat(pieces, dim=0)
        # A rank that received no tokens still has to keep ``permute_tokens`` on the
        # graph, or its backward never runs and the all-to-all is skipped on one rank.
        return output + permute_tokens * 0


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    *,
    num_experts: int,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """Routed SwiGLU with one shared LoRA pair per logical spec.

    Under expert parallelism this dispatches through the same all-to-all pipeline the
    fused rows use; the adapters are replicated, so only the base weights are sharded.
    """
    from .....distributed.moe import dispatch_to_ep_class, expert_parallel_size

    ep_size = expert_parallel_size()
    if ep_size > 1:
        # Same validation the non-LoRA reference row does: an EP state with weights
        # that were never sharded would be indexed with a local expert count against a
        # global routing map, so name the problem instead of raising a shape mismatch.
        if num_experts % ep_size or fc1_1_2_weight.shape[0] * ep_size != num_experts:
            raise ValueError(
                "MoE-LoRA (shared, eager) received expert-parallel weights that are not "
                f"sharded as expected (local {fc1_1_2_weight.shape[0]} vs global "
                f"{num_experts} over ep_size {ep_size}). Enable expert parallelism so "
                "tokens are dispatched to local experts."
            )
        return dispatch_to_ep_class(
            _SharedLoRAExpertMathEP,
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_2_weight,
            fc2_weight,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            lora_scale_gate,
            lora_scale_up,
            lora_scale_down,
        )

    scale_gate = hidden_states.new_tensor(lora_scale_gate)
    scale_up = hidden_states.new_tensor(lora_scale_up)
    scale_down = hidden_states.new_tensor(lora_scale_down)

    gate_delta = F.linear(F.linear(hidden_states, lora_a_gate), lora_b_gate) * scale_gate
    up_delta = F.linear(F.linear(hidden_states, lora_a_up), lora_b_up) * scale_up

    output = torch.zeros_like(hidden_states)
    for expert_idx, top_k_pos, token_idx in group_routing_assignments(selected_experts, num_experts):
        current_state = hidden_states[token_idx]
        mid = _shared_lora_mid(current_state, fc1_1_2_weight[expert_idx], gate_delta[token_idx], up_delta[token_idx])
        lora_x_down = F.linear(F.linear(mid, lora_a_down), lora_b_down) * scale_down
        current_hidden_states = F.linear(mid, fc2_weight[expert_idx]) + lora_x_down
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current_hidden_states.to(output.dtype))
    return output
