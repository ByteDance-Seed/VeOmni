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

"""CPU-runnable tests for the remaining eager ops behavior."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Eager MoE numerical-ordering guardrail (CPU-only)
# ---------------------------------------------------------------------------


def _eager_moe_routing_before_fc2(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Reference eager MoE with routing weights applied before fc2."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.silu(gate) * up
        y = y * routing_weights[token_idx, top_k_pos, None]
        y = F.linear(y, fc2_weight[idx])
        output.index_add_(0, token_idx, y.to(output.dtype))

    return output


def _eager_moe_routing_after_fc2(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Alternative eager MoE ordering used by the broken parity oracle."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.linear(F.silu(gate) * up, fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, y.to(output.dtype))

    return output


class TestEagerMoeOrdering:
    """CPU-only regression tests for the eager MoE parity oracle."""

    def test_routing_weight_order_matters_in_bf16(self):
        """The two mathematically equivalent orderings are not bf16-identical."""
        torch.manual_seed(0)

        num_tokens, num_experts, hidden_dim, ffn_dim, topk = 32, 4, 64, 32, 2
        dtype = torch.bfloat16

        hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, dtype=dtype)
        router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
        routing_weights = routing_weights.to(dtype)

        fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype)
        fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype)
        fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, dtype=dtype)

        before = _eager_moe_routing_before_fc2(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            swiglu_limit=0.1,
        )
        after = _eager_moe_routing_after_fc2(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            swiglu_limit=0.1,
        )

        assert not torch.equal(before, after)
