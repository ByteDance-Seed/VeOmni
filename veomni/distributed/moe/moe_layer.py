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
from .moe_utils import (
    generate_weights_idx,
    get_permutation_mapping,
    get_permuted_tokens_weight,
    inverse_sort_chunks_by_idxs,
    sort_chunks_by_idxs,
    unpermute,
)


if not is_torch_npu_available():
    from ...ops.kernels.moe._kernels.kernel.group_gemm import (
        group_gemm_same_mn,
        group_gemm_same_nk,
        group_gemm_same_nk_silu_gate_up,
    )


_POST_ALL2ALL_HIDDEN_CHUNK_SIZE = 32
_PRE_ALL2ALL_HIDDEN_CHUNK_SIZE = 128
_UNPERMUTE_CHUNK_SIZE = 8192
_EP_MERGED_GROUP_GEMM_TOKEN_CHUNK_SIZE = 2048


def _iter_token_chunks(total_tokens: int, chunk_size: int = _EP_MERGED_GROUP_GEMM_TOKEN_CHUNK_SIZE):
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        yield start, end


def _chunk_cumsum(cumsum: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return cumsum.clamp(min=start, max=end).sub(start)


def _all_to_all_no_autograd(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
) -> torch.Tensor:
    if dist.get_world_size(group=group) == 1:
        return input

    input = input.contiguous()
    output = torch.empty((sum(output_splits), input.size(1)), dtype=input.dtype, device=input.device)
    dist.all_to_all_single(
        output,
        input,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


class _TokenPreAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        local_input_permutation_mapping: torch.Tensor,
        num_global_tokens_per_local_expert: torch.Tensor,
        num_experts: int,
        input_splits: list[int],
        output_splits: list[int],
        ep_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        hidden_dim = hidden_states.size(-1)
        num_local_experts = num_experts // ep_group.size()
        permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
        split_sizes = num_global_tokens_per_local_expert.ravel()

        global_permuted_hidden_states = hidden_states.new_empty((sum(output_splits), hidden_dim))
        for start in range(0, hidden_dim, _PRE_ALL2ALL_HIDDEN_CHUNK_SIZE):
            end = min(start + _PRE_ALL2ALL_HIDDEN_CHUNK_SIZE, hidden_dim)
            local_permuted_hidden_states = hidden_states[:, start:end].index_select(0, local_input_permutation_mapping)
            global_permuted_chunk = _all_to_all_no_autograd(
                ep_group,
                local_permuted_hidden_states,
                output_splits,
                input_splits,
            )
            sort_chunks_by_idxs(
                global_permuted_chunk,
                split_sizes,
                permute_order,
                global_permuted_hidden_states[:, start:end],
            )

        ctx.num_experts = num_experts
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.ep_group = ep_group
        ctx.hidden_states_shape = hidden_states.shape
        ctx.permute_order = permute_order
        ctx.split_sizes = split_sizes
        ctx.save_for_backward(local_input_permutation_mapping)
        return global_permuted_hidden_states

    @staticmethod
    def backward(ctx, grad_global_permuted_hidden_states: torch.Tensor):
        (local_input_permutation_mapping,) = ctx.saved_tensors
        hidden_states_shape = ctx.hidden_states_shape
        hidden_dim = hidden_states_shape[-1]
        grad_hidden_states = torch.zeros(
            hidden_states_shape,
            device=grad_global_permuted_hidden_states.device,
            dtype=grad_global_permuted_hidden_states.dtype,
        )

        for start in range(0, hidden_dim, _PRE_ALL2ALL_HIDDEN_CHUNK_SIZE):
            end = min(start + _PRE_ALL2ALL_HIDDEN_CHUNK_SIZE, hidden_dim)
            grad_sorted_chunk = grad_global_permuted_hidden_states[:, start:end].contiguous()
            grad_global_permuted_chunk = inverse_sort_chunks_by_idxs(
                grad_sorted_chunk,
                ctx.split_sizes,
                ctx.permute_order,
            )
            grad_local_permuted_chunk = _all_to_all_no_autograd(
                ctx.ep_group,
                grad_global_permuted_chunk,
                ctx.input_splits,
                ctx.output_splits,
            )
            grad_hidden_states[:, start:end].scatter_add_(
                0,
                local_input_permutation_mapping.unsqueeze(1).expand(-1, end - start),
                grad_local_permuted_chunk,
            )

        return grad_hidden_states, None, None, None, None, None, None


def preprocess(
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)

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
    routing_map: torch.Tensor,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape

    local_input_permutation_mapping = get_permutation_mapping(hidden_states.size(0), routing_map)
    global_permuted_hidden_states = _TokenPreAllToAll.apply(
        hidden_states,
        local_input_permutation_mapping,
        num_global_tokens_per_local_expert,
        num_experts,
        input_splits,
        output_splits,
        ep_group,
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
    num_local_experts = num_experts // ep_group.size()
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    expert_output_splits = num_global_tokens_per_local_expert.T.ravel()

    hidden_dim = expert_outputs.size(-1)
    weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)
    if hidden_dim <= _POST_ALL2ALL_HIDDEN_CHUNK_SIZE:
        expert_outputs = sort_chunks_by_idxs(
            expert_outputs,
            expert_output_splits,
            unpermute_order,
        )
        unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)
        return unpermute(
            unpermute_outputs,
            weights_idx,
            org_hidden_states_shape,
            local_input_permutation_mapping,
            routing_map,
        )

    output = expert_outputs.new_empty(org_hidden_states_shape)
    for start in range(0, hidden_dim, _POST_ALL2ALL_HIDDEN_CHUNK_SIZE):
        end = min(start + _POST_ALL2ALL_HIDDEN_CHUNK_SIZE, hidden_dim)
        expert_output_chunk = sort_chunks_by_idxs(
            expert_outputs[:, start:end],
            expert_output_splits,
            unpermute_order,
        )
        chunk_outputs = all_to_all(ep_group, expert_output_chunk, input_splits, output_splits)
        output[:, start:end] = unpermute(
            chunk_outputs,
            weights_idx,
            torch.Size((org_hidden_states_shape[0], end - start)),
            local_input_permutation_mapping,
            routing_map,
        )

    return output


def _weighted_unpermute_forward(
    tokens: torch.Tensor,
    tokens_weight: torch.Tensor,
    permutation_mapping: torch.Tensor,
    output: torch.Tensor,
):
    hidden_dim = output.size(-1)
    output.zero_()
    for start in range(0, tokens.size(0), _UNPERMUTE_CHUNK_SIZE):
        end = min(start + _UNPERMUTE_CHUNK_SIZE, tokens.size(0))
        chunk_mapping = permutation_mapping[start:end]
        output.scatter_add_(
            0,
            chunk_mapping.unsqueeze(1).expand(-1, hidden_dim),
            (tokens[start:end] * tokens_weight[start:end].unsqueeze(-1)).to(output.dtype),
        )
    return output


def _routing_weight_indices(selected_experts: torch.Tensor, routing_map: torch.Tensor):
    expert_indices, token_indices = routing_map.nonzero(as_tuple=True)
    topk_match = selected_experts[token_indices] == expert_indices.unsqueeze(1)
    topk_indices = topk_match.to(torch.int64).argmax(dim=1)
    return token_indices, topk_indices


class EPMergedFc1PostAllToAllGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        routing_weights,
        selected_experts,
        routing_map,
        local_input_permutation_mapping,
        cumsum,
        fc1_1_2_weight,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        hidden_chunk_size: int,
        num_experts: int,
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        org_hidden_states_shape,
        ep_group,
        *fc2_weight_chunks,
    ):
        del fc2_weight_chunks
        hidden_dim = fc2_weight.shape[1]
        num_local_experts = num_experts // ep_group.size()
        unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
        expert_output_splits = num_global_tokens_per_local_expert.T.ravel()
        tokens_weight = get_permuted_tokens_weight(routing_weights, selected_experts, routing_map)

        output = permute_tokens.new_empty(org_hidden_states_shape)
        for hidden_start in range(0, hidden_dim, hidden_chunk_size):
            hidden_end = min(hidden_start + hidden_chunk_size, hidden_dim)
            fc2_weight_chunk = fc2_weight[:, hidden_start:hidden_end, :].contiguous()
            expert_output_chunk = permute_tokens.new_empty((permute_tokens.shape[0], hidden_end - hidden_start))
            for token_start, token_end in _iter_token_chunks(permute_tokens.shape[0]):
                chunk_cumsum = _chunk_cumsum(cumsum, token_start, token_end)
                fc1_result = group_gemm_same_nk_silu_gate_up(
                    a=permute_tokens[token_start:token_end],
                    b=fc1_1_2_weight,
                    cumsum_M=chunk_cumsum,
                    max_M=token_end - token_start,
                )
                expert_output_chunk[token_start:token_end].copy_(
                    group_gemm_same_nk(
                        a=fc1_result,
                        b=fc2_weight_chunk,
                        cumsum_M=chunk_cumsum,
                        max_M=token_end - token_start,
                        transpose_a=False,
                        transpose_b=True,
                    )
                )

            sorted_chunk = sort_chunks_by_idxs(expert_output_chunk, expert_output_splits, unpermute_order)
            local_chunk = _all_to_all_no_autograd(ep_group, sorted_chunk, input_splits, output_splits)
            _weighted_unpermute_forward(
                local_chunk,
                tokens_weight,
                local_input_permutation_mapping,
                output[:, hidden_start:hidden_end],
            )

        token_indices, topk_indices = _routing_weight_indices(selected_experts, routing_map)
        ctx.hidden_chunk_size = hidden_chunk_size
        ctx.num_fc2_weight_chunks = len(range(0, hidden_dim, hidden_chunk_size))
        ctx.num_experts = num_experts
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.ep_group = ep_group
        ctx.unpermute_order = unpermute_order
        ctx.expert_output_splits = expert_output_splits
        ctx.org_hidden_states_shape = org_hidden_states_shape
        ctx.save_for_backward(
            permute_tokens,
            routing_weights,
            token_indices,
            topk_indices,
            tokens_weight,
            local_input_permutation_mapping,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            routing_weights,
            token_indices,
            topk_indices,
            tokens_weight,
            local_input_permutation_mapping,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
        ) = ctx.saved_tensors
        hidden_chunk_size = ctx.hidden_chunk_size
        intermediate_dim = fc1_1_2_weight.shape[1] // 2
        fc2_weight_grad_arg_start = 17

        grad_routing_weights = torch.zeros_like(routing_weights) if ctx.needs_input_grad[1] else None
        grad_tokens_weight = torch.zeros_like(tokens_weight) if grad_routing_weights is not None else None

        grad_fc1_1_weight = None
        if ctx.needs_input_grad[7]:
            grad_fc1_1_weight = torch.zeros(
                (fc1_1_2_weight.shape[0], intermediate_dim, fc1_1_2_weight.shape[2]),
                dtype=fc1_1_2_weight.dtype,
                device=fc1_1_2_weight.device,
            )

        grad_fc1_2_weight = None
        if ctx.needs_input_grad[8]:
            grad_fc1_2_weight = torch.zeros(
                (fc1_1_2_weight.shape[0], intermediate_dim, fc1_1_2_weight.shape[2]),
                dtype=fc1_1_2_weight.dtype,
                device=fc1_1_2_weight.device,
            )

        fc2_weight_grads = [None] * ctx.num_fc2_weight_chunks
        for chunk_idx, hidden_start in enumerate(range(0, fc2_weight.shape[1], hidden_chunk_size)):
            hidden_end = min(hidden_start + hidden_chunk_size, fc2_weight.shape[1])
            if ctx.needs_input_grad[fc2_weight_grad_arg_start + chunk_idx]:
                fc2_weight_grads[chunk_idx] = torch.zeros(
                    (fc2_weight.shape[0], hidden_end - hidden_start, fc2_weight.shape[2]),
                    dtype=fc2_weight.dtype,
                    device=fc2_weight.device,
                )

        grad_fc1_result_all = torch.zeros(
            (permute_tokens.shape[0], intermediate_dim),
            dtype=permute_tokens.dtype,
            device=permute_tokens.device,
        )
        for chunk_idx, hidden_start in enumerate(range(0, fc2_weight.shape[1], hidden_chunk_size)):
            hidden_end = min(hidden_start + hidden_chunk_size, fc2_weight.shape[1])
            fc2_weight_chunk = fc2_weight[:, hidden_start:hidden_end, :].contiguous()
            grad_local = grad_output[:, hidden_start:hidden_end].index_select(0, local_input_permutation_mapping)

            expert_output_chunk = (
                permute_tokens.new_empty((permute_tokens.shape[0], hidden_end - hidden_start))
                if grad_tokens_weight is not None
                else None
            )

            grad_weighted_local = grad_local * tokens_weight.unsqueeze(-1)
            grad_sorted = _all_to_all_no_autograd(
                ctx.ep_group,
                grad_weighted_local.contiguous(),
                ctx.output_splits,
                ctx.input_splits,
            )
            grad_expert_full = inverse_sort_chunks_by_idxs(
                grad_sorted,
                ctx.expert_output_splits,
                ctx.unpermute_order,
            )

            grad_fc2_weight_chunk = fc2_weight_grads[chunk_idx]
            for token_start, token_end in _iter_token_chunks(permute_tokens.shape[0]):
                chunk_cumsum = _chunk_cumsum(cumsum, token_start, token_end)
                grad_expert_chunk = grad_expert_full[token_start:token_end].contiguous()

                if grad_fc2_weight_chunk is not None:
                    fc1_result = group_gemm_same_nk_silu_gate_up(
                        a=permute_tokens[token_start:token_end],
                        b=fc1_1_2_weight,
                        cumsum_M=chunk_cumsum,
                        max_M=token_end - token_start,
                    )
                    if expert_output_chunk is not None:
                        expert_output_chunk[token_start:token_end].copy_(
                            group_gemm_same_nk(
                                a=fc1_result,
                                b=fc2_weight_chunk,
                                cumsum_M=chunk_cumsum,
                                max_M=token_end - token_start,
                                transpose_a=False,
                                transpose_b=True,
                            )
                        )
                    group_gemm_same_mn(
                        a=grad_expert_chunk,
                        b=fc1_result,
                        c=grad_fc2_weight_chunk,
                        cumsum_K=chunk_cumsum,
                        max_K=token_end - token_start,
                        transpose_a=True,
                        transpose_b=False,
                        accumulate=True,
                    )

                elif expert_output_chunk is not None:
                    fc1_result = group_gemm_same_nk_silu_gate_up(
                        a=permute_tokens[token_start:token_end],
                        b=fc1_1_2_weight,
                        cumsum_M=chunk_cumsum,
                        max_M=token_end - token_start,
                    )
                    expert_output_chunk[token_start:token_end].copy_(
                        group_gemm_same_nk(
                            a=fc1_result,
                            b=fc2_weight_chunk,
                            cumsum_M=chunk_cumsum,
                            max_M=token_end - token_start,
                            transpose_a=False,
                            transpose_b=True,
                        )
                    )

                grad_fc1_result_all[token_start:token_end].add_(
                    group_gemm_same_nk(
                        a=grad_expert_chunk,
                        b=fc2_weight_chunk,
                        cumsum_M=chunk_cumsum,
                        max_M=token_end - token_start,
                        transpose_b=False,
                    )
                )

            if grad_tokens_weight is not None:
                sorted_output = sort_chunks_by_idxs(
                    expert_output_chunk,
                    ctx.expert_output_splits,
                    ctx.unpermute_order,
                )
                local_output = _all_to_all_no_autograd(
                    ctx.ep_group,
                    sorted_output,
                    ctx.input_splits,
                    ctx.output_splits,
                )
                grad_tokens_weight.add_((grad_local * local_output).sum(dim=-1).to(tokens_weight.dtype))

        grad_permute_tokens = permute_tokens
        for token_start, token_end in _iter_token_chunks(permute_tokens.shape[0]):
            chunk_cumsum = _chunk_cumsum(cumsum, token_start, token_end)
            permute_tokens_chunk = permute_tokens[token_start:token_end].clone()
            grad_fc1_result = grad_fc1_result_all[token_start:token_end]

            fc1_output = group_gemm_same_nk(
                a=permute_tokens_chunk,
                b=fc1_1_2_weight,
                cumsum_M=chunk_cumsum,
                max_M=token_end - token_start,
                transpose_a=False,
                transpose_b=True,
            )
            fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
            fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
            grad_fc1_2_output = fc1_1_activation * grad_fc1_result
            grad_fc1_1_activation = grad_fc1_result * fc1_2_output
            grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

            if grad_fc1_1_weight is not None:
                group_gemm_same_mn(
                    a=grad_fc1_1_output,
                    b=permute_tokens_chunk,
                    c=grad_fc1_1_weight,
                    cumsum_K=chunk_cumsum,
                    max_K=token_end - token_start,
                    transpose_a=True,
                    transpose_b=False,
                    accumulate=True,
                )
            if grad_fc1_2_weight is not None:
                group_gemm_same_mn(
                    a=grad_fc1_2_output,
                    b=permute_tokens_chunk,
                    c=grad_fc1_2_weight,
                    cumsum_K=chunk_cumsum,
                    max_K=token_end - token_start,
                    transpose_a=True,
                    transpose_b=False,
                    accumulate=True,
                )

            grad_fc1_output = torch.empty_like(fc1_output)
            grad_fc1_output[:, :intermediate_dim].copy_(grad_fc1_1_output)
            grad_fc1_output[:, intermediate_dim:].copy_(grad_fc1_2_output)
            grad_permute_tokens[token_start:token_end].copy_(
                group_gemm_same_nk(
                    a=grad_fc1_output,
                    b=fc1_1_2_weight,
                    cumsum_M=chunk_cumsum,
                    max_M=token_end - token_start,
                    transpose_b=False,
                )
            )

        if grad_routing_weights is not None:
            grad_routing_weights[token_indices, topk_indices] = grad_tokens_weight

        return (
            grad_permute_tokens,
            grad_routing_weights,
            None,
            None,
            None,
            None,
            None,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *fc2_weight_grads,
        )


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
        fc1_output = fc1_1_activation.mul_(fc1_2_output)
        del fc1_1_output, fc1_2_output

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

        ctx.save_for_backward(permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight)

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [tokens, hidden_dim]
        permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight = ctx.saved_tensors
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]

        fc1_1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )
        fc1_2_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

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


class EPMergedFc1GroupGemm(torch.autograd.Function):
    """EP autograd function that accepts a merged fc1_1_2 weight [E, 2I, H].

    Uses a single group_gemm_same_nk call for fc1 instead of two separate calls.
    """

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_2_weight,
        fc2_weight,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]
        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )

        fc2_output = permute_tokens.new_empty((permute_tokens.shape[0], fc2_weight.shape[1]))
        for start, end in _iter_token_chunks(permute_tokens.shape[0]):
            chunk_cumsum = _chunk_cumsum(cumsum, start, end)
            fc1_result = group_gemm_same_nk_silu_gate_up(
                a=permute_tokens[start:end],
                b=fc1_1_2_weight,
                cumsum_M=chunk_cumsum,
                max_M=end - start,
            )

            fc2_output[start:end].copy_(
                group_gemm_same_nk(
                    a=fc1_result,
                    b=fc2_weight,
                    cumsum_M=chunk_cumsum,
                    max_M=end - start,
                    transpose_a=False,
                    transpose_b=True,
                )
            )

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
        ) = ctx.saved_tensors

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            grad_fc2_weight.zero_()

        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            grad_fc1_1_2_weight.zero_()

        grad_permute_tokens = torch.empty_like(permute_tokens)
        for start, end in _iter_token_chunks(grad_output.shape[0]):
            chunk_cumsum = _chunk_cumsum(cumsum, start, end)
            permute_tokens_chunk = permute_tokens[start:end]
            grad_output_chunk = grad_output[start:end]

            fc1_result = group_gemm_same_nk_silu_gate_up(
                a=permute_tokens_chunk,
                b=fc1_1_2_weight,
                cumsum_M=chunk_cumsum,
                max_M=end - start,
            )

            if grad_fc2_weight is not None:
                group_gemm_same_mn(
                    a=grad_output_chunk,
                    b=fc1_result,
                    c=grad_fc2_weight,
                    cumsum_K=chunk_cumsum,
                    max_K=end - start,
                    transpose_a=True,
                    transpose_b=False,
                    accumulate=True,
                )

            grad_fc1_result = group_gemm_same_nk(
                a=grad_output_chunk,
                b=fc2_weight,
                cumsum_M=chunk_cumsum,
                max_M=end - start,
                transpose_b=False,
            )

            fc1_output = group_gemm_same_nk(
                a=permute_tokens_chunk,
                b=fc1_1_2_weight,
                cumsum_M=chunk_cumsum,
                max_M=end - start,
                transpose_a=False,
                transpose_b=True,
            )
            fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
            fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

            grad_fc1_2_output = fc1_1_activation * grad_fc1_result
            grad_fc1_1_activation = grad_fc1_result * fc1_2_output
            grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
            grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

            grad_permute_tokens[start:end].copy_(
                group_gemm_same_nk(
                    a=grad_fc1_output,
                    b=fc1_1_2_weight,
                    cumsum_M=chunk_cumsum,
                    max_M=end - start,
                    transpose_b=False,
                )
            )

            if grad_fc1_1_2_weight is not None:
                group_gemm_same_mn(
                    a=grad_fc1_output,
                    b=permute_tokens_chunk,
                    c=grad_fc1_1_2_weight,
                    cumsum_K=chunk_cumsum,
                    max_K=end - start,
                    transpose_a=True,
                    transpose_b=False,
                    accumulate=True,
                )

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
        )
