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


def require_top_k_entries(num_entries: int, num_tokens: int, top_k: int) -> None:
    """Refuse an entry count that cannot be ``num_tokens`` groups of ``top_k``.

    A shape comparison, so it costs nothing and needs no device sync -- and it catches
    what actually goes wrong in practice: a ``top_k`` that disagrees with the router, or
    a buffer that lost entries in transit. It does not, and cannot cheaply, establish
    that the entries are spread *evenly* over the tokens. That comes from the routing
    itself: ``torch.topk`` returns distinct experts, so no token can appear more than
    ``top_k`` times, and a total of ``num_tokens * top_k`` then leaves each token exactly
    ``top_k``. A router that drops tokens or repeats an expert would break that argument
    before it broke this check.
    """
    if num_entries != num_tokens * top_k:
        raise ValueError(
            f"the token permutation should carry exactly top_k={top_k} entries per token, but it covers "
            f"{num_entries} where {num_tokens} tokens need {num_tokens * top_k}. Grouping by token, which is "
            "what makes this sum reproducible, would pair entries with the wrong token."
        )


def sum_entries_per_token(
    values: torch.Tensor,
    permutation_mapping: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Add up each token's ``top_k`` entries over an axis rather than over atomics.

    Both directions of the expert-parallel permutation need this sum: the unpermute
    adds a token's ``top_k`` expert outputs, and the permute's backward adds the
    gradient of its ``top_k`` copies. The obvious spelling of either -- ``scatter_add_``
    on the token index, or letting autograd answer a duplicated ``index_select`` -- has
    ``top_k`` duplicate destinations, which CUDA serves with atomics, so the same input
    adds up in a different order on every call.

    Measured on the 26B-A4B recipe that is not the last bit it sounds like. Two forwards
    of one batch first diverged at an expert output by a single bf16 step, and the
    residual stack turned that into 0.25 one layer later; the backward is coarser by six
    orders of magnitude, because the gradient is bf16 and every atomic add rounds to it,
    so eight values summed in a different order landed up to 1.25e-1 apart -- a few
    percent of the gradient, on 614 of 631 parameters. Between two runs of identical
    data and frozen weights, ``grad_norm`` ranged over 380 to 3300.

    Do not try to reproduce that from one call. The unpermute's atomics accumulate in
    FP32 and disagree by ~1e-6, which the cast back to bf16 discards, so in isolation the
    old forward repeats bit for bit at every bfloat16 shape tried; only the backward,
    which accumulated in bf16 itself, drifts visibly on a single call. What the model has
    and one call does not is scale. ``scripts/bench_moe_token_permutation.py`` measures
    both.

    ``use_deterministic_algorithms`` does not catch any of this: only the 1-D CUDA form
    of ``scatter_add_`` raises, and the ``index_add`` behind a gather's backward is
    reached through autograd rather than through a checked op.

    Grouping an entry with the others belonging to its token makes the order of
    accumulation a property of the shape, and accumulating in FP32 matches the non-EP
    ``moe_gather`` kernel, which has always summed top-k in a fixed order in FP32. The
    sort has to be stable for that order to belong to the shape rather than to the sort's
    tie-breaking, which nothing promises to keep and no test can observe.

    Only correct because top-k routing gives every token exactly ``top_k`` distinct
    experts, so the entry count pins the grouping -- checked, because being one entry off
    would pair entries with the wrong token and stay silent about it.
    """
    require_top_k_entries(permutation_mapping.numel(), num_tokens, top_k)
    require_top_k_entries(values.shape[0], num_tokens, top_k)

    ordered = torch.argsort(permutation_mapping, stable=True)
    grouped = values[ordered].view(num_tokens, top_k, values.shape[-1])
    return grouped.sum(dim=1, dtype=torch.float32).to(values.dtype)


class _PermuteTokens(torch.autograd.Function):
    """A gather forward, whose backward sums instead of accumulating with atomics.

    ``index_select`` on indices carrying ``top_k`` duplicates is a deterministic read,
    but autograd inverts it with an ``index_add`` into the token buffer, and that is
    where the order stops being fixed. The gradient of a token is the sum of its
    ``top_k`` copies either way; this spells that sum out.
    """

    @staticmethod
    def forward(ctx, tokens: torch.Tensor, permutation_mapping: torch.Tensor, top_k: int) -> torch.Tensor:
        ctx.save_for_backward(permutation_mapping)
        ctx.num_tokens, ctx.top_k = tokens.shape[0], top_k
        return tokens.index_select(0, permutation_mapping)

    @staticmethod
    def backward(ctx, grad_entries: torch.Tensor):
        (permutation_mapping,) = ctx.saved_tensors
        grad_tokens = sum_entries_per_token(grad_entries, permutation_mapping, ctx.num_tokens, ctx.top_k)
        return grad_tokens, None, None


def permute(tokens: torch.Tensor, routing_map: torch.Tensor, top_k: int):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].
        top_k (int): How many experts each token is routed to.

    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # The backward checks this too, but reaching it from there means every rank has
    # already run the expert forward and its all-to-alls. Raising in the middle of that
    # leaves the other ranks in a collective, so the job hangs to a watchdog timeout
    # instead of reporting the mismatch.
    require_top_k_entries(sorted_indices.numel(), num_tokens, top_k)

    # use the mapping to permute the tokens
    permuted_input = _PermuteTokens.apply(tokens, sorted_indices, top_k)

    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
    top_k: int,
):
    """
    Unpermutes the tokens and apply the weight.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens * top_k, hidden_dim].
        routing_weights (torch.Tensor): The routing weights, [num_tokens, num_experts].
        hidden_states_shape (torch.Size): The shape of the hidden states, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].
        top_k (int): How many experts each token was routed to.

    Returns:
        torch.Tensor: The unpermuted token tensor, [num_tokens, hidden_dim].
    """
    require_top_k_entries(tokens.shape[0], hidden_states_shape[0], top_k)

    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    return sum_entries_per_token(tokens, permutation_mapping, hidden_states_shape[0], top_k)


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
