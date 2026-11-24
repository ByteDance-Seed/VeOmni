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
    """Refuse an entry count no token could have produced under ``top_k`` slots.

    A shape comparison, so it costs nothing and needs no device sync -- and it catches
    what actually goes wrong in practice: a ``top_k`` that disagrees with the router, or
    a buffer that lost entries in transit. The bound is an upper one rather than an
    equality because hash routing (DeepSeek V4) can select the same expert twice, and
    that repeated slot collapses to one entry while its weights are summed into the same
    expert. So a token carries between one and ``top_k`` entries, and only a count outside
    that range contradicts the routing. Entries are still grouped by token, which is what
    makes the sum reproducible; the group sizes just need not be uniform.
    """
    if not num_tokens <= num_entries <= num_tokens * top_k:
        raise ValueError(
            f"a token permutation of top_k={top_k} over {num_tokens} tokens covers {num_tokens}.."
            f"{num_tokens * top_k} entries, but this one has {num_entries}."
        )


def sum_entries_per_token(
    values: torch.Tensor,
    permutation_mapping: torch.Tensor,
    num_tokens: int,
    top_k: int | None = None,
) -> torch.Tensor:
    """Add up each token's entries over an axis rather than over atomics.

    Both directions of the expert-parallel permutation need this sum: the unpermute
    adds a token's expert outputs, and the permute's backward adds the gradient of its
    copies. The obvious spelling of either -- ``scatter_add_`` on the token index, or
    letting autograd answer a duplicated ``index_select`` -- has several duplicate
    destinations per token, which CUDA serves with atomics, so the same input adds up
    in a different order on every call.

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

    Top-k routing normally gives every token exactly ``top_k`` distinct experts, but a
    hash router (DeepSeek V4) may repeat one, and a repeated slot is a single entry in
    ``permutation_mapping`` while its weight is summed into the same expert. The grouping
    therefore uses each token's real entry count; ``top_k`` only pins that count down when
    no slot repeats, so it is checked when given rather than relied on.
    """
    entries_per_token = torch.bincount(permutation_mapping, minlength=num_tokens)
    if top_k is not None:
        require_top_k_entries(permutation_mapping.numel(), num_tokens, top_k)
        require_top_k_entries(values.shape[0], num_tokens, top_k)
    elif bool((entries_per_token == 0).any()):
        raise ValueError(
            f"every one of {num_tokens} tokens needs at least one entry in the permutation, got counts "
            f"{entries_per_token.tolist()[:16]} for {permutation_mapping.numel()} entries."
        )

    ordered_indices = torch.argsort(permutation_mapping, stable=True)
    # `segment_reduce` sums each token's contiguous run in a fixed order, so uneven run
    # lengths (a repeated hash slot) need no reshape and no padding. It has no dtype
    # argument, so the FP32 accumulation has to come from the cast before it.
    return torch.segment_reduce(values[ordered_indices].float(), "sum", lengths=entries_per_token, axis=0).to(
        values.dtype
    )


class _PermuteTokens(torch.autograd.Function):
    """A gather forward, whose backward sums instead of accumulating with atomics.

    ``index_select`` on indices carrying ``top_k`` duplicates is a deterministic read,
    but autograd inverts it with an ``index_add`` into the token buffer, and that is
    where the order stops being fixed. The gradient of a token is the sum of its
    ``top_k`` copies either way; this spells that sum out.
    """

    @staticmethod
    def forward(ctx, tokens: torch.Tensor, permutation_mapping: torch.Tensor, top_k: int | None) -> torch.Tensor:
        ctx.save_for_backward(permutation_mapping)
        ctx.num_tokens, ctx.top_k = tokens.shape[0], top_k
        return tokens.index_select(0, permutation_mapping)

    @staticmethod
    def backward(ctx, grad_entries: torch.Tensor):
        (permutation_mapping,) = ctx.saved_tensors
        grad_tokens = sum_entries_per_token(grad_entries, permutation_mapping, ctx.num_tokens, ctx.top_k)
        return grad_tokens, None, None


def permute(tokens: torch.Tensor, routing_map: torch.Tensor, top_k: int | None = None):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].
        top_k (int | None): How many experts each token is routed to. Pass it to assert the
            routing width; leave it ``None`` when the router may repeat an expert (hash
            routing), where the real entry count is taken from ``routing_map`` instead.

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
    if top_k is not None:
        require_top_k_entries(sorted_indices.numel(), num_tokens, top_k)
    elif bool((torch.bincount(sorted_indices, minlength=num_tokens) == 0).any()):
        raise ValueError(
            f"every one of {num_tokens} routed tokens needs at least one expert entry, got "
            f"{sorted_indices.numel()} entries for {num_tokens} tokens."
        )

    # use the mapping to permute the tokens
    permuted_input = _PermuteTokens.apply(tokens, sorted_indices, top_k)

    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
    top_k: int | None = None,
):
    """
    Unpermutes the tokens and apply the weight.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens * top_k, hidden_dim].
        routing_weights (torch.Tensor): The routing weights, [num_tokens, num_experts].
        hidden_states_shape (torch.Size): The shape of the hidden states, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].
        top_k (int | None): How many experts each token was routed to. Passing it selects the
            grouped, order-independent sum below; ``None`` keeps the fused kernel dispatch.

    Returns:
        torch.Tensor: The unpermuted token tensor, [num_tokens, hidden_dim].
    """
    out_dtype = tokens.dtype
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())

    if top_k is not None:
        # The routing weights may be FP32 under FSDP2 bf16 compute, which promotes the
        # product; the caller expects the expert outputs back in the activation dtype.
        require_top_k_entries(tokens.shape[0], hidden_states_shape[0], top_k)
        weighted = tokens * tokens_weight.unsqueeze(-1)
        return sum_entries_per_token(weighted, permutation_mapping, hidden_states_shape[0], top_k).to(out_dtype)

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
