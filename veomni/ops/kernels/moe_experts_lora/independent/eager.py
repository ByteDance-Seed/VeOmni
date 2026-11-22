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

"""independent MoE-LoRA eager math. Regular autograd, no custom backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..routing import group_routing_assignments


def _independent_lora_mid(
    state: Tensor,
    fc1_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
) -> Tensor:
    """``act(gate) * up`` for one expert, with that expert's own LoRA pairs folded in."""
    gate, up = F.linear(state, fc1_weight).chunk(2, dim=-1)
    gate_hidden = F.linear(state, lora_a_gate)
    up_hidden = F.linear(state, lora_a_up)
    gate = torch.addmm(gate, gate_hidden, lora_b_gate.T, alpha=lora_scale_gate)
    up = torch.addmm(up, up_hidden, lora_b_up.T, alpha=lora_scale_up)
    return F.silu(gate) * up


class _IndependentLoRAExpertMathEP:
    """Per-expert-LoRA math on tokens already dispatched by EP.

    ``cumsum`` bounds the rows per local expert, and the 3-D adapter tensors are
    sharded along their leading expert dim exactly like the base weights, so a local
    expert index reads its own slice. Routing weights are applied afterwards by
    ``tokens_post_all2all``.
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
        """Run ``down(act(gate) * up)`` plus each local expert's own LoRA deltas.

        Routing weights are applied later by ``tokens_post_all2all``.
        """
        scale_down = permute_tokens.new_tensor(lora_scale_down)

        num_local = fc1_1_2_weight.shape[0]
        ends = cumsum.to(dtype=torch.long)
        starts = permute_tokens.new_zeros(num_local, dtype=torch.long)
        if num_local > 1:
            starts[1:] = ends[:-1]
        pieces: list[Tensor] = []
        for expert_idx in range(num_local):
            start = int(starts[expert_idx])
            end = int(ends[expert_idx])
            mid = _independent_lora_mid(
                permute_tokens[start:end],
                fc1_1_2_weight[expert_idx],
                lora_a_gate[expert_idx],
                lora_b_gate[expert_idx],
                lora_a_up[expert_idx],
                lora_b_up[expert_idx],
                lora_scale_gate,
                lora_scale_up,
            )
            lora_delta_down = F.linear(F.linear(mid, lora_a_down[expert_idx]), lora_b_down[expert_idx]) * scale_down
            current = F.linear(mid, fc2_weight[expert_idx]) + lora_delta_down
            pieces.append(current.to(permute_tokens.dtype))
        output = torch.cat(pieces, dim=0)
        # Keep ``permute_tokens`` on the graph so an empty rank still runs the
        # all-to-all backward.
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
    """Routed SwiGLU with a per-expert LoRA pair per logical spec.

    Under expert parallelism the adapters are sharded alongside the base weights, so
    this dispatches through the same pipeline the fused rows use.
    """
    from .....distributed.moe import dispatch_to_ep_class, expert_parallel_size

    ep_size = expert_parallel_size()
    if ep_size > 1:
        # As in the shared variant and the non-LoRA reference row: verify the weights
        # really are sharded before indexing them with local expert ids.
        if num_experts % ep_size or fc1_1_2_weight.shape[0] * ep_size != num_experts:
            raise ValueError(
                "MoE-LoRA (independent, eager) received expert-parallel weights that are "
                f"not sharded as expected (local {fc1_1_2_weight.shape[0]} vs global "
                f"{num_experts} over ep_size {ep_size}). Enable expert parallelism so "
                "tokens are dispatched to local experts."
            )
        return dispatch_to_ep_class(
            _IndependentLoRAExpertMathEP,
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

    scale_down = hidden_states.new_tensor(lora_scale_down)

    output = torch.zeros_like(hidden_states)
    for expert_idx, top_k_pos, token_idx in group_routing_assignments(selected_experts, num_experts):
        mid = _independent_lora_mid(
            hidden_states[token_idx],
            fc1_1_2_weight[expert_idx],
            lora_a_gate[expert_idx],
            lora_b_gate[expert_idx],
            lora_a_up[expert_idx],
            lora_b_up[expert_idx],
            lora_scale_gate,
            lora_scale_up,
        )
        lora_x_down = F.linear(F.linear(mid, lora_a_down[expert_idx]), lora_b_down[expert_idx]) * scale_down
        current_hidden_states = F.linear(mid, fc2_weight[expert_idx]) + lora_x_down
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current_hidden_states.to(output.dtype))
    return output
