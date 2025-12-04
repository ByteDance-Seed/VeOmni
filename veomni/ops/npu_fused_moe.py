import torch
import torch_npu
from mindspeed.core.fusions.grouped_matmul import Ops
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute


def npu_group_gemm(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    weights = weights.transpose(1, 2)
    out = Ops.gmm(x, weights, split_sizes, trans_b=False)
    return out


def npu_fused_moe_forward(
    module,
    num_experts,
    routing_weights,
    selected_experts,
    hidden_states,
    fc1_1_weight,
    fc1_2_weight,
    fc2_weight,
):
    # permute
    permuted_hidden_states, row_ids_map = npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)
    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])

    tokens_per_expert_group = tokens_per_expert_group.view(1, -1)

    fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()
    fc1_output = npu_group_gemm(permuted_hidden_states, fc1_weight, tokens_per_expert)
    fc1_activation = torch_npu.npu_swiglu(fc1_output, dim=-1)
    fc2_out = npu_group_gemm(fc1_activation, fc2_weight, tokens_per_expert)

    # unpermute
    output = npu_moe_token_unpermute(fc2_out, row_ids_map, probs=routing_weights)
    return output
