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


def compact_permute(tokens: torch.Tensor, selected_experts: torch.Tensor):
    """Stable expert-major dispatch without a dense expert/token mask.

    Native top-k and its replica aliases have unique physical IDs per token.
    Keep the flattened occurrence index so combine can recover its exact weight.
    """
    topk = selected_experts.shape[-1]
    if topk == 0:
        raise ValueError("Compact EP dispatch requires at least one selected expert.")
    order = torch.argsort(selected_experts.reshape(-1), stable=True)
    token_indices = torch.div(order, topk, rounding_mode="floor")
    return tokens.index_select(0, token_indices), order


def compact_unpermute(tokens, routing_weights, hidden_states_shape, occurrence_order):
    """Preserve expert-major multiplication and the existing FP32 scatter sum."""
    weights = routing_weights.reshape(-1).index_select(0, occurrence_order)
    token_indices = torch.div(occurrence_order, routing_weights.shape[-1], rounding_mode="floor")
    from ...ops.kernels.moe_experts.shared.unpermute import eager

    return eager(tokens, weights, token_indices, num_tokens=hidden_states_shape[0])


def alias_permute(tokens: torch.Tensor, physical_experts: torch.Tensor, logical_experts: torch.Tensor):
    """Dispatch physical rows while accumulating input gradients in logical order.

    The second gather is a one-to-one occurrence permutation, so its backward
    cannot add or round contributions. The first gather retains the unbalanced
    dispatch's logical expert-major accumulation order, independent of aliases.
    """
    if physical_experts.shape != logical_experts.shape:
        raise ValueError("Physical aliases must preserve the logical routing shape.")
    topk = physical_experts.shape[-1]
    if topk == 0:
        raise ValueError("Alias EP dispatch requires at least one selected expert.")
    logical_order = torch.argsort(logical_experts.reshape(-1), stable=True)
    physical_order = torch.argsort(physical_experts.reshape(-1), stable=True)
    inverse_logical = torch.empty_like(logical_order)
    positions = torch.arange(logical_order.numel(), device=logical_order.device)
    inverse_logical.scatter_(0, logical_order, positions)
    physical_to_logical = inverse_logical.index_select(0, physical_order)
    logical_to_physical = torch.empty_like(physical_to_logical)
    logical_to_physical.scatter_(0, physical_to_logical, positions)
    logical_tokens = tokens.index_select(0, torch.div(logical_order, topk, rounding_mode="floor"))
    return logical_tokens.index_select(0, physical_to_logical), (logical_order, logical_to_physical)


def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


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
    out_dtype = tokens.dtype
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())

    from ...ops import VeomniOp
    from ...utils.device import IS_CUDA_AVAILABLE, get_device_type

    impl = (
        "triton"
        if IS_CUDA_AVAILABLE
        and torch.version.hip is None
        and tokens.device.type == get_device_type()
        and not torch.are_deterministic_algorithms_enabled()
        else "eager"
    )
    return VeomniOp("moe_unpermute", "standard", impl)(
        tokens, tokens_weight, permutation_mapping, num_tokens=hidden_states_shape[0]
    ).to(out_dtype)


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


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output
