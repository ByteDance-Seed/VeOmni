import torch
import torch.nn as nn

class Experts(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_size, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_size),
            requires_grad=True,
        )
        self.act_fn = nn.ReLU

    def forward(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None):
        if expert_idx is not None:
            assert not get_parallel_state().ep_enabled, "_moe_implementation=`eager` does not support EP"
            gate_proj_out = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
            up_proj_out = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))

            out = self.act_fn(gate_proj_out) * up_proj_out
            out = torch.matmul(out, self.down_proj[expert_idx].transpose(0, 1))
        else:
            assert routing_weights is not None and selected_experts is not None, (
                "routing_weights and selected_experts must be provided when expert_idx is None"
            )

            out = fused_moe_forward(
                module=self,
                num_experts=self.num_experts,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_proj,
                fc1_2_weight=self.up_proj,
                fc2_weight=self.down_proj,
            )
        return out
class MoeBlock(nn.Module):

