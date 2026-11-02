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


"""Backend-neutral balanced-EP validation and physical telemetry."""

from typing import Any

import torch
import torch.distributed as dist

from ..parallel_state import get_parallel_state
from .ep_load_balance import EPBalancePlan, _validate_executor_plan


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
    num_experts: int,
    selected_experts: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    load_balancer: Any,
) -> EPBalancePlan:
    """Validate shared balanced-EP metadata before any replica transfer."""
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
    if physical_ids.is_cuda:
        # Keep the full defensive validation, but avoid one GPU synchronization
        # for every range/count scalar read below.
        physical_ids = physical_ids.cpu()
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
    return plan


def _record_ep_balance(load_balancer: Any, plan: EPBalancePlan) -> None:
    from ...utils.moe_monitor import get_active_monitor

    monitor = get_active_monitor()
    record_ep_balance = getattr(monitor, "record_ep_balance", None)
    # Plans already contain all EP senders. The monitor later sums across the
    # FSDP/DP+SP group, which includes EP siblings in the supported topology.
    # Contribute each replicated plan once, not once per EP rank.
    if callable(record_ep_balance) and dist.get_rank(load_balancer.ep_group) == 0:
        record_ep_balance(
            load_balancer.layer_index,
            plan.rank_loads_before,
            plan.rank_loads_after,
            len(plan.replicas),
            sum(replica.moved_tokens for replica in plan.replicas),
        )
