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


from typing import Optional

import torch
import torch.distributed as dist

from ...utils.import_utils import is_torch_npu_available
from .comm import all_to_all
from .moe_utils import generate_weights_idx, permute, sort_chunks_by_idxs, unpermute


if not is_torch_npu_available():
    from ...ops.group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk


def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))

    # [ep_size] represent the number of sum tokens in each rank
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()

    # gather all the number of tokens per expert from all ep ranks
    # [ep_size, num_experts]
    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    # [ep_size, num_local_experts]
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    # [ep_size]
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    # [num_local_expert]
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        torch.device("cpu"), non_blocking=True
    )

    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(
        torch.device("cpu"), non_blocking=True
    )

    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = permute(hidden_states, routing_map)

    global_permuted_hidden_states = all_to_all(ep_group, local_permuted_hidden_states, output_splits, input_splits)

    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    global_permuted_hidden_states = sort_chunks_by_idxs(
        global_permuted_hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )

    return global_permuted_hidden_states, routing_map, local_input_permutation_mapping, org_hidden_states_shape


def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: int,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    # [tokens, experts]
    weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)

    unpermute_outputs = unpermute(
        unpermute_outputs,
        weights_idx,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
    )

    return unpermute_outputs


class EPGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]

        # compute linear layer fc1-1
        fc1_1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # compute linear layer fc1-2
        fc1_2_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # compute the actication of linear layer fc1-1
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # compute final result of linear layer fc1
        fc1_output = fc1_1_activation * fc1_2_output

        # weighted projection is outside this function
        # compute linear layer fc2
        fc2_output = group_gemm_same_nk(
            a=fc1_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [tokens, hidden_dim]
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        ) = ctx.saved_tensors
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]

        # dgrad fc1
        grad_fc1_output = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        # wgrad fc2
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_output,
                b=fc1_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output

        # dgrad output 2
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad fc1-2
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=permute_tokens,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # dgrad output 1
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad fc1-1
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=permute_tokens,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # grad input
        grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )
