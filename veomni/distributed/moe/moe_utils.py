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


import torch


def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    """
    sorted_indices = get_permutation_mapping(tokens.size(0), routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def get_permutation_mapping(num_tokens: int, routing_map: torch.Tensor):
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    return token_indices.masked_select(routing_map)


def build_routing_map(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    num_tokens, top_k = selected_experts.shape
    routing_map = torch.zeros(
        (num_experts, num_tokens),
        dtype=torch.bool,
        device=selected_experts.device,
    )
    token_indices = torch.arange(num_tokens, device=selected_experts.device).unsqueeze(0).expand(top_k, -1)
    routing_map[selected_experts.T.reshape(-1), token_indices.reshape(-1)] = True
    return routing_map


def get_permuted_tokens_weight(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    expert_indices, token_indices = routing_map.nonzero(as_tuple=True)
    topk_match = selected_experts[token_indices] == expert_indices.unsqueeze(1)
    topk_indices = topk_match.to(torch.int64).argmax(dim=1)
    return routing_weights[token_indices, topk_indices]


def unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
):
    """
    Unpermutes the tokens and apply the weight.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_weights (torch.Tensor): The routing weights, [num_tokens, num_experts].
        hidden_states_shape (torch.Size): The shape of the hidden states, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    Returns:
        torch.Tensor: The unpermuted token tensor, [num_tokens, hidden_dim].
    """
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())

    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]

    # Accumulate in FP32 to match the non-EP moe_gather kernel which also uses FP32
    # accumulation internally, reducing top-k rounding error from BF16 scatter_add_.
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=torch.float32)
    unpermuted_tokens.scatter_add_(0, permutation_mapping.unsqueeze(1).expand(-1, hidden_dim), tokens.float())
    return unpermuted_tokens.to(tokens.dtype)


def generate_weights_idx(routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts) -> torch.Tensor:
    """
    Generate the weight index for the unpermute operation.

    Args:
        routing_weights (torch.Tensor): The routing weights. shape [num_tokens, topk].
        selected_experts (torch.Tensor): The selected experts. shape [num_tokens, topk].
        num_experts (int): The number of experts. shape [num_tokens, num_experts].

    Returns:
        torch.Tensor: The weight index.
    """
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device)

    weights_idx.scatter_add_(1, selected_experts, routing_weights)

    return weights_idx


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor, output=None):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    split_sizes = split_sizes.tolist()
    if output is None:
        output = torch.empty_like(input)

    input_offsets = [0]
    for split_size in split_sizes:
        input_offsets.append(input_offsets[-1] + split_size)

    output_offset = 0
    for idx in sorted_idxs:
        split_size = split_sizes[idx]
        if split_size > 0:
            input_start = input_offsets[idx]
            input_end = input_start + split_size
            output_end = output_offset + split_size
            output[output_offset:output_end].copy_(input[input_start:input_end])
            output_offset = output_end
    return output


def inverse_sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor, output=None
):
    """Undo sort_chunks_by_idxs for the same split sizes and sorted indices."""
    split_sizes = split_sizes.tolist()
    if output is None:
        output = torch.empty_like(input)

    input_offsets = [0]
    for split_size in split_sizes:
        input_offsets.append(input_offsets[-1] + split_size)

    sorted_input_offset = 0
    for idx in sorted_idxs:
        split_size = split_sizes[idx]
        if split_size > 0:
            input_start = sorted_input_offset
            input_end = input_start + split_size
            output_start = input_offsets[idx]
            output_end = output_start + split_size
            output[output_start:output_end].copy_(input[input_start:input_end])
            sorted_input_offset = input_end
    return output
