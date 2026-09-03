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


from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from ..parallel_state import get_parallel_state
from .comm import all_to_all
from .moe_utils import generate_weights_idx, permute, sort_chunks_by_idxs, unpermute


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


def dispatch_to_ep_class(
    ep_class: Callable[..., torch.Tensor],
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    *ep_class_args: Any,
) -> torch.Tensor:
    """Shared EP MoE plumbing: ``preprocess`` + ``token_pre_all2all`` + ``ep_class.apply`` + ``tokens_post_all2all``.

    Mirrors the EP branch of every fused MoE forward (``group_gemm_fused_moe_forward``,
    ``group_gemm_fused_lora_moe_forward``, ``group_gemm_fused_independent_lora_moe_forward``,
    Quack equivalents): the all-to-all dispatch + cumsum computation + combine
    pattern is identical regardless of which autograd ``ep_class`` is being
    called. Extracting it here keeps each call site to a single line and lets
    the EP plumbing evolve in lock-step across LoRA / non-LoRA paths.

    Args:
        ep_class: Caller-owned EP autograd ``Function``. Typical classes live
            in ``veomni.kernels._kernels.moe_experts`` or ``moe_experts_lora``.
            ``ep_class.apply`` must accept ``(permute_tokens, cumsum, *ep_class_args)``
            in that order and return a ``[T_local, H]`` permuted-output tensor.
        num_experts: total expert count ``E`` for this MoE layer (global on EP).
        routing_weights: ``[B*S, topk]`` per-(token, slot) routing weights —
            applied later by ``tokens_post_all2all`` → ``unpermute``.
        selected_experts: ``[B*S, topk]`` per-(token, slot) global expert ids.
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        *ep_class_args: extra positional args forwarded to ``ep_class.apply``
            after ``permute_tokens`` and ``cumsum`` — typically the per-rank
            slice of base weights (and, for LoRA variants, LoRA tensors and
            scales).

    Returns:
        ``[B, S, H]`` (or ``[N, H]``) — same shape as ``hidden_states``.
    """
    ep_state = get_parallel_state()
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        preprocess(expert_mask=expert_mask, num_experts=num_experts, ep_group=ep_state.ep_group)
    )
    permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
        hidden_states=hidden_states,
        expert_mask=expert_mask,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
        ep_group=ep_state.ep_group,
    )
    cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)

    final_permute_tokens = ep_class.apply(permute_tokens, cumsum, *ep_class_args)

    return tokens_post_all2all(
        expert_outputs=final_permute_tokens,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
        routing_map=routing_map,
        local_input_permutation_mapping=local_input_permutation_mapping,
        org_hidden_states_shape=org_hidden_states_shape,
        ep_group=ep_state.ep_group,
    )


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
