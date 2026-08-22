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

from ...utils.import_utils import is_torch_npu_available
from ..parallel_state import get_parallel_state
from .comm import all_to_all
from .ep_load_balance import (
    EPBalancePlan,
    ExpertReplicaTransfer,
    _validate_executor_plan,
    cat_local_and_replica_weights,
)
from .moe_utils import generate_weights_idx, permute, sort_chunks_by_idxs, unpermute


if not is_torch_npu_available():
    from ...ops.kernels.moe._kernels.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk


def _apply_swiglu_clamp(fc1_1_output, fc1_2_output, swiglu_limit):
    """gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation.

    Returns ``(fc1_1_clamped, fc1_2_clamped, mask_fc1_1, mask_fc1_2)``.
    No-op (and returns ``None`` masks) when ``swiglu_limit is None``.
    Mirrors the helper in ``ops/kernels/moe/group_gemm.py`` to keep this
    module free of circular imports.
    """
    if swiglu_limit is None:
        return fc1_1_output, fc1_2_output, None, None
    mask_fc1_1 = fc1_1_output <= swiglu_limit
    mask_fc1_2 = (fc1_2_output >= -swiglu_limit) & (fc1_2_output <= swiglu_limit)
    fc1_1_output = fc1_1_output.clamp(max=swiglu_limit)
    fc1_2_output = fc1_2_output.clamp(min=-swiglu_limit, max=swiglu_limit)
    return fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2


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


def _validate_merged_expert_weights(gate_up_proj: Any, down_proj: Any) -> None:
    error = "EP load balancing requires compatible merged expert weights"
    if not isinstance(gate_up_proj, torch.Tensor) or gate_up_proj.ndim != 3:
        raise ValueError(f"{error}: gate_up_proj must be a 3-D tensor.")
    if not isinstance(down_proj, torch.Tensor) or down_proj.ndim != 3:
        raise ValueError(f"{error}: down_proj must be a 3-D tensor.")
    if gate_up_proj.shape[0] != down_proj.shape[0]:
        raise ValueError(f"{error}: gate_up_proj and down_proj must have the same expert-row count.")
    if gate_up_proj.shape[1] % 2 != 0:
        raise ValueError(f"{error}: gate_up_proj.shape[1] must be even.")
    if gate_up_proj.shape[1] != 2 * down_proj.shape[2]:
        raise ValueError(f"{error}: gate_up_proj.shape[1] must equal 2 * down_proj.shape[2].")
    if gate_up_proj.shape[2] != down_proj.shape[1]:
        raise ValueError(f"{error}: gate_up_proj.shape[2] must equal down_proj.shape[1].")
    if gate_up_proj.dtype != down_proj.dtype:
        raise ValueError(f"{error}: gate_up_proj and down_proj must have the same dtype.")
    if gate_up_proj.device != down_proj.device:
        raise ValueError(f"{error}: gate_up_proj and down_proj must be on the same device.")
    if gate_up_proj.layout != torch.strided or down_proj.layout != torch.strided:
        raise ValueError(f"{error}: gate_up_proj and down_proj must use the dense strided layout.")


def _validate_ep_balance_dispatch(
    ep_class: Callable[..., torch.Tensor],
    num_experts: int,
    selected_experts: torch.Tensor,
    ep_class_args: tuple[Any, ...],
    load_balancer: Any,
) -> tuple[EPBalancePlan, torch.Tensor, torch.Tensor, Any]:
    if ep_class is not EPMergedFc1GroupGemm:
        raise ValueError("EP load balancing supports only the merged-fc1 EPMergedFc1GroupGemm path.")
    if len(ep_class_args) != 3:
        raise ValueError("Balanced merged EP dispatch expects exactly (gate_up_proj, down_proj, swiglu_limit).")
    gate_up_proj, down_proj, swiglu_limit = ep_class_args
    _validate_merged_expert_weights(gate_up_proj, down_proj)
    if not callable(getattr(load_balancer, "build_plan", None)):
        raise TypeError("load_balancer must expose build_plan(selected_experts).")

    ep_state = get_parallel_state()
    ep_group = getattr(load_balancer, "ep_group", None)
    if ep_group is None or ep_group is not ep_state.ep_group:
        raise ValueError("load_balancer.ep_group must be the active expert-parallel process group.")

    plan = load_balancer.build_plan(selected_experts)
    if not isinstance(plan, EPBalancePlan):
        raise TypeError("load_balancer.build_plan must return an EPBalancePlan.")
    if plan.ep_size <= 1 or num_experts <= 0 or num_experts % plan.ep_size != 0:
        raise ValueError("The balance plan EP size must be greater than one and divide num_experts.")
    if plan.num_experts != num_experts:
        raise ValueError(f"The balance plan has {plan.num_experts} logical experts, expected {num_experts}.")
    if plan.num_local_experts != num_experts // plan.ep_size:
        raise ValueError("The balance plan local expert count is inconsistent with its logical expert count.")
    if tuple(plan.selected_physical_experts.shape) != tuple(selected_experts.shape):
        raise ValueError("The balance plan physical expert IDs must preserve selected_experts shape.")
    if plan.selected_physical_experts.device != selected_experts.device:
        raise ValueError("The balance plan physical expert IDs must stay on the selected_experts device.")
    if plan.selected_physical_experts.dtype != torch.long:
        raise TypeError("The balance plan physical expert IDs must use torch.long.")

    physical_slots_per_rank = plan.num_local_experts + (plan.max_replicas_per_rank if plan.replicas else 0)
    physical_num_experts = plan.ep_size * physical_slots_per_rank if plan.replicas else num_experts
    physical_ids = plan.selected_physical_experts.reshape(-1)
    if physical_ids.numel() and (
        int(physical_ids.min().item()) < 0 or int(physical_ids.max().item()) >= physical_num_experts
    ):
        raise ValueError(f"The balance plan physical expert IDs must be in [0, {physical_num_experts}).")

    physical_counts = plan.tokens_per_local_physical_expert
    expected_counts_shape = (plan.ep_size, physical_slots_per_rank)
    if tuple(physical_counts.shape) != expected_counts_shape:
        raise ValueError(
            "The balance plan physical count matrix has shape "
            f"{tuple(physical_counts.shape)}, expected {expected_counts_shape}."
        )
    if physical_counts.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("The balance plan physical count matrix must contain integers.")
    if bool((physical_counts < 0).any()):
        raise ValueError("The balance plan physical count matrix cannot contain negative values.")
    if not isinstance(plan.input_splits, tuple) or not isinstance(plan.output_splits, tuple):
        raise TypeError("The balance plan input and output splits must be exact tuples.")
    if len(plan.input_splits) != plan.ep_size or len(plan.output_splits) != plan.ep_size:
        raise ValueError("The balance plan input and output splits must each contain one entry per EP rank.")
    if any(not isinstance(split, int) or split < 0 for split in (*plan.input_splits, *plan.output_splits)):
        raise ValueError("The balance plan input and output splits must contain non-negative integers.")
    if sum(plan.input_splits) != selected_experts.numel():
        raise ValueError("The balance plan input splits do not conserve local routing occurrences.")
    if sum(plan.output_splits) != int(physical_counts.sum().item()):
        raise ValueError("The balance plan output splits do not match its local physical count matrix.")
    expected_input_splits = tuple(
        int(value)
        for value in torch.bincount(physical_ids, minlength=physical_num_experts)
        .reshape(plan.ep_size, physical_slots_per_rank)
        .sum(dim=1)
        .tolist()
    )
    expected_output_splits = tuple(int(value) for value in physical_counts.sum(dim=1).tolist())
    if plan.input_splits != expected_input_splits or plan.output_splits != expected_output_splits:
        raise ValueError("The balance plan input or output splits do not match its physical routing metadata.")
    if len(plan.rank_loads_before) != plan.ep_size or len(plan.rank_loads_after) != plan.ep_size:
        raise ValueError("The balance plan rank-load telemetry must contain one entry per EP rank.")

    # Validate both local weights before starting either P2P stream. In
    # particular, a malformed down projection must not be discovered after a
    # gate/up send has already been issued.
    _validate_executor_plan(gate_up_proj, plan, ep_group)
    _validate_executor_plan(down_proj, plan, ep_group)
    return plan, gate_up_proj, down_proj, swiglu_limit


def _record_ep_balance(load_balancer: Any, plan: EPBalancePlan) -> None:
    from ...utils.moe_monitor import get_active_monitor

    monitor = get_active_monitor()
    record_ep_balance = getattr(monitor, "record_ep_balance", None)
    if callable(record_ep_balance):
        record_ep_balance(
            load_balancer.layer_index,
            plan.rank_loads_before,
            plan.rank_loads_after,
            len(plan.replicas),
            sum(replica.moved_tokens for replica in plan.replicas),
        )


def _dispatch_to_balanced_ep(
    ep_class: Callable[..., torch.Tensor],
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    ep_class_args: tuple[Any, ...],
    load_balancer: Any,
) -> torch.Tensor:
    plan, gate_up_proj, down_proj, swiglu_limit = _validate_ep_balance_dispatch(
        ep_class,
        num_experts,
        selected_experts,
        ep_class_args,
        load_balancer,
    )
    _record_ep_balance(load_balancer, plan)

    ep_group = load_balancer.ep_group
    gate_up_transfer = None
    down_transfer = None
    if plan.replicas:
        gate_up_transfer = ExpertReplicaTransfer.start(gate_up_proj, plan, ep_group)
        down_transfer = ExpertReplicaTransfer.start(down_proj, plan, ep_group)

    physical_slots_per_rank = plan.tokens_per_local_physical_expert.shape[1]
    physical_num_experts = plan.ep_size * physical_slots_per_rank if plan.replicas else num_experts
    selected_physical_experts = plan.selected_physical_experts
    expert_mask = torch.nn.functional.one_hot(
        selected_physical_experts,
        num_classes=physical_num_experts,
    ).permute(2, 1, 0)
    permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
        hidden_states=hidden_states,
        expert_mask=expert_mask,
        num_experts=physical_num_experts,
        input_splits=plan.input_splits,
        output_splits=plan.output_splits,
        num_global_tokens_per_local_expert=plan.tokens_per_local_physical_expert,
        ep_group=ep_group,
    )

    if gate_up_transfer is not None and down_transfer is not None:
        replica_gate_up_proj = gate_up_transfer.wait()
        replica_down_proj = down_transfer.wait()
        gate_up_proj = cat_local_and_replica_weights(gate_up_proj, replica_gate_up_proj, plan, ep_group)
        down_proj = cat_local_and_replica_weights(down_proj, replica_down_proj, plan, ep_group)

    physical_counts = plan.tokens_per_local_physical_expert.sum(dim=0)
    cumsum = torch.cumsum(physical_counts, dim=0).to(permute_tokens.device)
    final_permute_tokens = EPMergedFc1GroupGemm.apply(
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        swiglu_limit,
    )

    return tokens_post_all2all(
        expert_outputs=final_permute_tokens,
        routing_weights=routing_weights,
        selected_experts=selected_physical_experts,
        num_experts=physical_num_experts,
        input_splits=plan.input_splits,
        output_splits=plan.output_splits,
        num_global_tokens_per_local_expert=plan.tokens_per_local_physical_expert,
        routing_map=routing_map,
        local_input_permutation_mapping=local_input_permutation_mapping,
        org_hidden_states_shape=org_hidden_states_shape,
        ep_group=ep_group,
    )


def dispatch_to_ep_class(
    ep_class: Callable[..., torch.Tensor],
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    *ep_class_args: Any,
    load_balancer: Any = None,
) -> torch.Tensor:
    """Shared EP MoE plumbing: ``preprocess`` + ``token_pre_all2all`` + ``ep_class.apply`` + ``tokens_post_all2all``.

    Mirrors the EP branch of every fused MoE forward (``group_gemm_fused_moe_forward``,
    ``group_gemm_fused_lora_moe_forward``, ``group_gemm_fused_independent_lora_moe_forward``,
    Quack equivalents): the all-to-all dispatch + cumsum computation + combine
    pattern is identical regardless of which autograd ``ep_class`` is being
    called. Extracting it here keeps each call site to a single line and lets
    the EP plumbing evolve in lock-step across LoRA / non-LoRA paths.

    Args:
        ep_class: The EP autograd ``Function`` class to apply (one of
            :class:`EPGroupGemm`, :class:`EPMergedFc1GroupGemm`, or one of the
            LoRA variants in ``veomni.lora.ops.moe_group_gemm``).
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
        load_balancer: optional private Qwen3.5 EP controller. When present,
            dispatch uses its immutable physical-alias plan and accepts only
            the merged gate/up and down projection argument layout.

    Returns:
        ``[B, S, H]`` (or ``[N, H]``) — same shape as ``hidden_states``.
    """
    if load_balancer is not None:
        return _dispatch_to_balanced_ep(
            ep_class,
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            ep_class_args,
            load_balancer,
        )

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


class EPGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit=None,
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

        # gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation. No-op when
        # swiglu_limit is None (legacy MoE models) — masks are None and the
        # ``if swiglu_limit is not None`` guards in backward are skipped.
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
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

        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
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
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
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

        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

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
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)

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
            None,  # swiglu_limit
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
        swiglu_limit=None,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]
        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )

        # Single fc1 gemm: output shape [T, 2I]
        fc1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # chunk is a view, no copy
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        # gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation. ``_apply_swiglu_clamp``
        # creates new tensors when ``swiglu_limit is not None`` so the saved halves are
        # independent of ``fc1_output`` storage; otherwise it is a no-op.
        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        # compute the activation of linear layer fc1-1
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # compute final result of linear layer fc1
        fc1_result = fc1_1_activation * fc1_2_output

        # compute linear layer fc2
        fc2_output = group_gemm_same_nk(
            a=fc1_result,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        # dgrad fc2
        grad_fc1_result = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad fc2
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_output,
                b=fc1_result,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # gate gradients
        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        # Merge grads back to [T, 2I]
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

        # single dgrad for merged fc1
        grad_permute_tokens = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # single wgrad for merged fc1
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_output,
                b=permute_tokens,
                c=grad_fc1_1_2_weight,
                cumsum_K=cumsum,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
            None,  # swiglu_limit
        )
