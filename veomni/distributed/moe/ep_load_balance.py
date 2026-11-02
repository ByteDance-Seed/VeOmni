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

import heapq
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from torch import nn

from ..parallel_state import get_parallel_state
from .ep_buffer import ReplicaBufferLease, ReplicaBufferPool


__all__ = [
    "EPBalancePlan",
    "ExpertReplica",
    "ExpertReplicaTransfer",
    "attach_qwen3_5_moe_ep_load_balancers",
    "build_ep_balance_plan",
    "cat_local_and_replica_weights",
]


@dataclass(frozen=True)
class ExpertReplica:
    """One temporary physical copy of a logical expert."""

    source_expert: int
    source_rank: int
    source_local_expert: int
    target_rank: int
    target_slot: int
    alias_expert: int
    moved_tokens: int


@dataclass(frozen=True)
class EPBalancePlan:
    """Immutable metadata and local rewrite for one EP forward."""

    ep_size: int
    num_experts: int
    num_local_experts: int
    max_replicas_per_rank: int
    selected_physical_experts: torch.Tensor
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    tokens_per_local_physical_expert: torch.Tensor
    replicas: tuple[ExpertReplica, ...]
    rank_loads_before: tuple[int, ...]
    rank_loads_after: tuple[int, ...]


def _validate_executor_plan(
    local_weight: torch.Tensor,
    plan: EPBalancePlan,
    ep_group: dist.ProcessGroup,
) -> int:
    if not isinstance(local_weight, torch.Tensor):
        raise TypeError("local_weight must be a torch.Tensor.")
    if local_weight.layout != torch.strided:
        raise ValueError("local_weight must use a dense strided layout.")
    if local_weight.ndim < 1:
        raise ValueError("local_weight must have an expert-row dimension.")
    if not isinstance(plan, EPBalancePlan):
        raise TypeError("plan must be an EPBalancePlan.")
    if ep_group is None:
        raise ValueError("Expert replica transfer requires a real expert-parallel process group.")

    ep_size = dist.get_world_size(group=ep_group)
    ep_rank = dist.get_rank(group=ep_group)
    if plan.ep_size != ep_size:
        raise ValueError(f"plan.ep_size ({plan.ep_size}) does not match the process-group size ({ep_size}).")
    if plan.ep_size <= 1:
        raise ValueError("Expert replica transfer requires ep_size greater than one.")
    if plan.num_local_experts <= 0 or plan.num_experts != plan.ep_size * plan.num_local_experts:
        raise ValueError("The plan expert counts are inconsistent with its EP size.")
    if local_weight.shape[0] != plan.num_local_experts:
        raise ValueError(f"local_weight has {local_weight.shape[0]} expert rows, expected {plan.num_local_experts}.")
    if not 0 < plan.max_replicas_per_rank <= plan.num_local_experts:
        raise ValueError("plan.max_replicas_per_rank must be positive and no greater than num_local_experts.")

    used_target_slots: set[tuple[int, int]] = set()
    stride = plan.num_local_experts + plan.max_replicas_per_rank
    for replica in plan.replicas:
        if not isinstance(replica, ExpertReplica):
            raise TypeError("plan.replicas must contain only ExpertReplica entries.")
        if not 0 <= replica.source_rank < plan.ep_size or not 0 <= replica.target_rank < plan.ep_size:
            raise ValueError("Replica source and target ranks must be valid EP-local ranks.")
        if replica.source_rank == replica.target_rank:
            raise ValueError("A replica source and target rank must differ.")
        if not 0 <= replica.source_local_expert < plan.num_local_experts:
            raise ValueError("Replica source_local_expert is outside the local expert row range.")
        if replica.source_expert != replica.source_rank * plan.num_local_experts + replica.source_local_expert:
            raise ValueError("Replica source metadata does not map to its source-local expert row.")
        if not 0 <= replica.target_slot < plan.max_replicas_per_rank:
            raise ValueError("Replica target_slot is outside the fixed replica slot range.")
        target_slot = (replica.target_rank, replica.target_slot)
        if target_slot in used_target_slots:
            raise ValueError("Multiple replicas cannot occupy the same target rank and slot.")
        used_target_slots.add(target_slot)
        expected_alias = replica.target_rank * stride + plan.num_local_experts + replica.target_slot
        if replica.alias_expert != expected_alias:
            raise ValueError("Replica alias_expert does not match its target rank and slot.")

    return ep_rank


class ExpertReplicaTransfer:
    """An in-flight asynchronous transfer into fixed temporary replica slots."""

    def __init__(
        self,
        replica_weight: torch.Tensor,
        work_handles: tuple[dist.Work, ...],
        communication_tensors: tuple[torch.Tensor, ...],
        *,
        lease: ReplicaBufferLease | None = None,
    ) -> None:
        self._replica_weight = replica_weight
        self._work_handles = work_handles
        self._communication_tensors = communication_tensors
        self._completed = False
        self._lease = lease
        self._failed = False

    @classmethod
    def start(
        cls,
        local_weight: torch.Tensor,
        plan: EPBalancePlan,
        ep_group: dist.ProcessGroup,
        *,
        pool: ReplicaBufferPool | None = None,
    ) -> "ExpertReplicaTransfer":
        """Launch all plan-required weight sends and receives without waiting."""
        ep_rank = _validate_executor_plan(local_weight, plan, ep_group)
        shape = (plan.max_replicas_per_rank, *local_weight.shape[1:])
        lease = pool.acquire(local_weight, shape) if pool is not None else None
        replica_weight = lease.tensor if lease is not None else local_weight.new_zeros(shape)
        work_handles: list[dist.Work] = []
        communication_tensors: list[torch.Tensor] = []

        for replica in plan.replicas:
            if ep_rank == replica.source_rank:
                send_tensor = local_weight[replica.source_local_expert].detach().contiguous()
                target_global_rank = dist.get_global_rank(ep_group, replica.target_rank)
                work = dist.isend(send_tensor, dst=target_global_rank, group=ep_group)
                communication_tensors.append(send_tensor)
                work_handles.append(work)
            elif ep_rank == replica.target_rank:
                recv_tensor = replica_weight[replica.target_slot]
                source_global_rank = dist.get_global_rank(ep_group, replica.source_rank)
                work = dist.irecv(recv_tensor, src=source_global_rank, group=ep_group)
                communication_tensors.append(recv_tensor)
                work_handles.append(work)

        return cls(replica_weight, tuple(work_handles), tuple(communication_tensors), lease=lease)

    def wait(self) -> torch.Tensor:
        """Wait for every P2P operation and return all fixed replica slots."""
        if self._failed:
            raise RuntimeError("Cannot wait on a failed replica transfer.")
        if not self._completed:
            try:
                for work in self._work_handles:
                    work.wait()
                if self._lease is not None:
                    self._lease.record_use()
            except Exception:
                self._failed = True
                raise
            self._completed = True
        return self._replica_weight

    def record_stream_use(self) -> None:
        """Record a compute consumer after submission; retain this transfer with its graph."""
        if not self._completed:
            raise RuntimeError("Call wait before recording replica compute use.")
        if self._lease is not None:
            self._lease.record_use()


def _validate_replica_weight(
    local_weight: torch.Tensor,
    replica_weight: torch.Tensor,
    plan: EPBalancePlan,
    ep_group: dist.ProcessGroup,
) -> None:
    _validate_executor_plan(local_weight, plan, ep_group)
    if not isinstance(replica_weight, torch.Tensor):
        raise TypeError("replica_weight must be a torch.Tensor.")
    if isinstance(replica_weight, nn.Parameter):
        raise TypeError("replica_weight must be a temporary tensor, not an nn.Parameter.")
    expected_shape = (plan.max_replicas_per_rank, *local_weight.shape[1:])
    if tuple(replica_weight.shape) != expected_shape:
        raise ValueError(f"replica_weight has shape {tuple(replica_weight.shape)}, expected {expected_shape}.")
    if replica_weight.dtype != local_weight.dtype:
        raise TypeError("replica_weight and local_weight must have the same dtype.")
    if replica_weight.device != local_weight.device:
        raise ValueError("replica_weight and local_weight must be on the same device.")
    if replica_weight.layout != local_weight.layout:
        raise ValueError("replica_weight and local_weight must have the same layout.")


class _CatLocalAndReplicaWeights(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_weight: torch.Tensor,
        replica_weight: torch.Tensor,
        plan: EPBalancePlan,
        ep_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.plan = plan
        ctx.ep_group = ep_group
        return torch.cat((local_weight, replica_weight), dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return _merge_replica_weight_gradients(grad_output, ctx.plan, ctx.ep_group), None, None, None


def _merge_replica_weight_gradients(
    grad_output: torch.Tensor, plan: EPBalancePlan, ep_group: dist.ProcessGroup
) -> torch.Tensor:
    return _merge_replica_weight_gradient_parts(
        grad_output[: plan.num_local_experts], grad_output[plan.num_local_experts :], plan, ep_group
    )


def _merge_replica_weight_gradient_parts(
    local_grad: torch.Tensor,
    replica_grad: torch.Tensor,
    plan: EPBalancePlan,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Return replica contributions in the existing order, without mutating borrowed gradients."""
    ep_rank = dist.get_rank(group=ep_group)
    local_grad = local_grad.clone()
    work_handles: list[dist.Work] = []
    communication_tensors: list[torch.Tensor] = []
    received_gradients: list[tuple[int, torch.Tensor]] = []
    for replica in plan.replicas:
        if ep_rank == replica.target_rank:
            send_tensor = replica_grad[replica.target_slot].contiguous()
            owner_global_rank = dist.get_global_rank(ep_group, replica.source_rank)
            work_handles.append(dist.isend(send_tensor, dst=owner_global_rank, group=ep_group))
            communication_tensors.append(send_tensor)
        elif ep_rank == replica.source_rank:
            # Sender and receiver must agree on contiguous logical row layout.
            recv_tensor = torch.empty_like(
                local_grad[replica.source_local_expert], memory_format=torch.contiguous_format
            )
            target_global_rank = dist.get_global_rank(ep_group, replica.target_rank)
            work_handles.append(dist.irecv(recv_tensor, src=target_global_rank, group=ep_group))
            communication_tensors.append(recv_tensor)
            received_gradients.append((replica.source_local_expert, recv_tensor))
    for work in work_handles:
        work.wait()
    for source_local_expert, received_gradient in received_gradients:
        local_grad[source_local_expert].add_(received_gradient)
    return local_grad


def cat_local_and_replica_weights(
    local_weight: torch.Tensor,
    replica_weight: torch.Tensor,
    plan: EPBalancePlan,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Expose original expert rows followed by fixed temporary replica slots."""
    _validate_replica_weight(local_weight, replica_weight, plan, ep_group)
    if not plan.replicas:
        return local_weight
    return _CatLocalAndReplicaWeights.apply(local_weight, replica_weight, plan, ep_group)


@dataclass
class _ReplicaState:
    source_expert: int
    source_rank: int
    target_rank: int
    target_slot: int
    moved_tokens: int


def _rank_load_profile(loads: list[int]) -> tuple[int, ...]:
    """Prioritize the busiest ranks without getting stuck on tied extrema."""
    return tuple(sorted(loads, reverse=True))


def _waterfill_allocations(total_tokens: int, base_loads: dict[int, int]) -> dict[int, int]:
    """Allocate integer tokens to the lowest loaded ranks with rank-id ties."""
    if total_tokens < 0 or not base_loads:
        raise ValueError("Water-filling requires non-negative tokens and at least one rank.")

    ordered = sorted(base_loads, key=lambda rank: (base_loads[rank], rank))
    active = [ordered[0]]
    level = base_loads[ordered[0]]
    remaining = total_tokens
    cursor = 1

    while cursor < len(ordered):
        next_level = base_loads[ordered[cursor]]
        cost = (next_level - level) * len(active)
        if remaining < cost:
            break
        remaining -= cost
        level = next_level
        while cursor < len(ordered) and base_loads[ordered[cursor]] == level:
            active.append(ordered[cursor])
            cursor += 1

    quotient, remainder = divmod(remaining, len(active))
    final_level = level + quotient
    remainder_ranks = set(sorted(active)[:remainder])
    allocations = dict.fromkeys(base_loads, 0)
    for rank in active:
        allocations[rank] = final_level - base_loads[rank] + int(rank in remainder_ranks)

    if sum(allocations.values()) != total_tokens:
        raise RuntimeError("Water-filling failed to conserve expert tokens.")
    return allocations


def _validate_planner_inputs(
    selected_experts: torch.Tensor,
    tokens_per_expert_by_rank: torch.Tensor,
    num_experts: int,
    ep_rank: int,
    max_replicas_per_rank: int,
) -> tuple[int, int, torch.Tensor]:
    if not isinstance(selected_experts, torch.Tensor):
        raise TypeError("selected_experts must be a torch.Tensor.")
    if selected_experts.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("selected_experts must contain integer expert IDs.")
    if tokens_per_expert_by_rank.ndim != 2:
        raise ValueError("tokens_per_expert_by_rank must have shape [ep_size, num_experts].")

    ep_size, gathered_num_experts = tokens_per_expert_by_rank.shape
    if num_experts <= 0 or gathered_num_experts != num_experts:
        raise ValueError("num_experts must match the gathered histogram width and be positive.")
    if ep_size <= 1:
        raise ValueError("EP load balancing requires ep_size greater than one.")
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}.")

    num_local_experts = num_experts // ep_size
    if max_replicas_per_rank <= 0:
        raise ValueError("max_replicas_per_rank must be positive.")
    if max_replicas_per_rank > num_local_experts:
        raise ValueError(
            "max_replicas_per_rank must be no greater than num_local_experts "
            f"({num_local_experts}), got {max_replicas_per_rank}."
        )

    counts = tokens_per_expert_by_rank.to(device="cpu", dtype=torch.long).contiguous()
    if bool((counts < 0).any()):
        raise ValueError("Gathered expert token counts must be non-negative.")

    flat_selected = selected_experts.reshape(-1)
    if flat_selected.numel() and (
        int(flat_selected.min().item()) < 0 or int(flat_selected.max().item()) >= num_experts
    ):
        raise ValueError(f"selected_experts IDs must be in [0, {num_experts}).")
    local_counts = torch.bincount(flat_selected.to(device="cpu", dtype=torch.long), minlength=num_experts)
    if not torch.equal(local_counts, counts[ep_rank]):
        raise ValueError("The local selected_experts histogram does not match the gathered EP-rank row.")

    return ep_size, num_local_experts, counts


def _plan_replicas(
    counts: torch.Tensor,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    max_replicas_per_rank: int,
) -> tuple[list[_ReplicaState], list[int], list[int], list[dict[int, int]]]:
    expert_totals = [int(value) for value in counts.sum(dim=0).tolist()]
    rank_loads_before = [
        sum(expert_totals[rank * num_local_experts : (rank + 1) * num_local_experts]) for rank in range(ep_size)
    ]
    rank_loads = rank_loads_before.copy()
    expert_allocations = [{expert // num_local_experts: total} for expert, total in enumerate(expert_totals)]
    replica_states: list[_ReplicaState] = []
    replica_indices_by_expert: list[list[int]] = [[] for _ in range(num_experts)]
    used_slots: list[set[int]] = [set() for _ in range(ep_size)]
    owners = [expert // num_local_experts for expert in range(num_experts)]
    owner_tokens = expert_totals.copy()
    excluded_targets = [1 << owner for owner in owners]
    available_targets = (1 << ep_size) - 1
    expert_order = [
        sorted(
            range(owner * num_local_experts, (owner + 1) * num_local_experts),
            key=lambda expert: (-owner_tokens[expert], expert),
        )
        for owner in range(ep_size)
    ]

    def candidate_keys(owner, minimum_target_load):
        if not available_targets & ~(1 << owner):
            return
        for expert in expert_order[owner]:
            if owner_tokens[expert] <= 0:
                break
            if not available_targets & ~excluded_targets[expert]:
                continue
            # Before an expert has replicas, a first move needs a load gap of
            # at least two. Using the lightest available rank is sufficient
            # for rejecting impossible candidates before building heap entries.
            if not replica_indices_by_expert[expert] and rank_loads[owner] <= minimum_target_load + 1:
                continue
            yield (-rank_loads[owner], -owner_tokens[expert], owner, expert)

    while True:
        # Integer loads that differ by at most one already attain the minimum
        # descending-sorted profile for this fixed token total. No replica can
        # strictly improve the objective, so avoid searching equivalent moves.
        if not available_targets or max(rank_loads) - min(rank_loads) <= 1:
            break
        target_order = sorted(
            (rank for rank in range(ep_size) if available_targets & (1 << rank)),
            key=lambda rank: (rank_loads[rank], rank),
        )
        minimum_target_load = rank_loads[target_order[0]]
        # Each owner list is sorted by the remaining complete priority key;
        # rank load is constant within that list for this decision. A lazy
        # merge preserves the original global order without materializing
        # every expert's tuple after every accepted move.
        candidate_experts = heapq.merge(*(candidate_keys(owner, minimum_target_load) for owner in range(ep_size)))

        accepted = None
        current_profile = _rank_load_profile(rank_loads)
        for _, _, owner, expert in candidate_experts:
            existing_indices = replica_indices_by_expert[expert]
            target_candidates = [rank for rank in target_order if not excluded_targets[expert] & (1 << rank)]

            for target in target_candidates:
                # A first replica only transfers integer tokens between two
                # ranks. If their load gap is at most one, a positive transfer
                # can only tie (swap the loads) or worsen the sorted profile.
                # Later targets are no lighter. Existing replicas may instead
                # redistribute among several ranks and must not be pruned.
                if not existing_indices and rank_loads[owner] - rank_loads[target] <= 1:
                    break
                locations = [owner, *(replica_states[index].target_rank for index in existing_indices), target]
                base_loads = {rank: rank_loads[rank] - expert_allocations[expert].get(rank, 0) for rank in locations}
                allocations = _waterfill_allocations(expert_totals[expert], base_loads)
                if allocations[target] <= 0 or any(
                    allocations[replica_states[index].target_rank] <= 0 for index in existing_indices
                ):
                    continue

                proposed_loads = rank_loads.copy()
                for rank in locations:
                    proposed_loads[rank] = base_loads[rank] + allocations[rank]
                # A tied maximum or minimum can keep the global spread fixed
                # while one hot rank improves. Compare all ranks, busiest
                # first, so independent hot/cold pairs can move in sequence.
                if _rank_load_profile(proposed_loads) >= current_profile:
                    continue

                accepted = (expert, owner, target, existing_indices, allocations, proposed_loads)
                break
            if accepted is not None:
                break

        if accepted is None:
            break

        expert, owner, target, existing_indices, allocations, rank_loads = accepted
        target_slot = next(slot for slot in range(max_replicas_per_rank) if slot not in used_slots[target])
        used_slots[target].add(target_slot)
        if len(used_slots[target]) == max_replicas_per_rank:
            available_targets &= ~(1 << target)
        excluded_targets[expert] |= 1 << target
        owner_tokens[expert] = allocations.get(owner, 0)
        # Only the chosen expert's owner allocation changed. Re-sort that
        # owner's small list; all other local priority orders remain valid.
        expert_order[owner].sort(key=lambda item: (-owner_tokens[item], item))
        replica_states.append(
            _ReplicaState(
                source_expert=expert,
                source_rank=owner,
                target_rank=target,
                target_slot=target_slot,
                moved_tokens=allocations[target],
            )
        )
        replica_indices_by_expert[expert].append(len(replica_states) - 1)
        for index in existing_indices:
            state = replica_states[index]
            state.moved_tokens = allocations[state.target_rank]
        expert_allocations[expert] = allocations

    return replica_states, rank_loads_before, rank_loads, expert_allocations


def _allocate_replica_rows(
    counts: torch.Tensor,
    replica_states: list[_ReplicaState],
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    # Counts have already been validated and transferred to the host. Keep
    # scalar bookkeeping in Python rather than dispatching one Tensor op per
    # replica/sender cell, then materialize the result once.
    remaining_values = counts.tolist()
    allocations_by_replica = []
    for state in replica_states:
        remaining = state.moved_tokens
        row_allocations = []
        for sender in range(counts.shape[0]):
            moved = min(remaining, remaining_values[sender][state.source_expert])
            remaining_values[sender][state.source_expert] -= moved
            row_allocations.append(moved)
            remaining -= moved
        if remaining != 0:
            raise RuntimeError("A replica requests more concrete occurrences than its source expert owns.")
        allocations_by_replica.append(tuple(row_allocations))
    remaining_counts = torch.tensor(remaining_values, dtype=counts.dtype, device=counts.device).reshape_as(counts)
    return remaining_counts, allocations_by_replica


def _build_ep_balance_plan_from_counts(
    selected_experts: torch.Tensor,
    tokens_per_expert_by_rank: torch.Tensor,
    num_experts: int,
    ep_rank: int,
    max_replicas_per_rank: int,
    *,
    token_granularity: int = 1,
) -> EPBalancePlan:
    """Pure count-driven planning helper used by the public collective entry point."""
    ep_size, num_local_experts, counts = _validate_planner_inputs(
        selected_experts,
        tokens_per_expert_by_rank,
        num_experts,
        ep_rank,
        max_replicas_per_rank,
    )
    replica_states, rank_loads_before, rank_loads_after, _ = _plan_replicas(
        counts,
        ep_size,
        num_experts,
        num_local_experts,
        max_replicas_per_rank,
    )

    if type(token_granularity) is not int or token_granularity <= 0:
        raise ValueError("token_granularity must be a positive integer.")
    if token_granularity > 1:
        replica_states = [
            replace(state, moved_tokens=state.moved_tokens // token_granularity * token_granularity)
            for state in replica_states
            if state.moved_tokens >= token_granularity
        ]
        rank_loads_after = rank_loads_before.copy()
        for state in replica_states:
            rank_loads_after[state.source_rank] -= state.moved_tokens
            rank_loads_after[state.target_rank] += state.moved_tokens
        # Quantization can remove useful outgoing moves from a rank that is
        # also a target. Never retain a rounded plan that worsens the objective.
        if _rank_load_profile(rank_loads_after) >= _rank_load_profile(rank_loads_before):
            replica_states = []
            rank_loads_after = rank_loads_before.copy()

    if not replica_states:
        start = ep_rank * num_local_experts
        local_counts = counts[:, start : start + num_local_experts].clone()
        input_splits = tuple(
            int(counts[ep_rank, rank * num_local_experts : (rank + 1) * num_local_experts].sum().item())
            for rank in range(ep_size)
        )
        output_splits = tuple(int(value) for value in local_counts.sum(dim=1).tolist())
        return EPBalancePlan(
            ep_size=ep_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            max_replicas_per_rank=max_replicas_per_rank,
            selected_physical_experts=selected_experts.to(dtype=torch.long).clone(),
            input_splits=input_splits,
            output_splits=output_splits,
            tokens_per_local_physical_expert=local_counts,
            replicas=(),
            rank_loads_before=tuple(rank_loads_before),
            rank_loads_after=tuple(rank_loads_after),
        )

    stride = num_local_experts + max_replicas_per_rank
    remaining_counts, allocations_by_replica = _allocate_replica_rows(counts, replica_states)
    replicas = tuple(
        ExpertReplica(
            source_expert=state.source_expert,
            source_rank=state.source_rank,
            source_local_expert=state.source_expert % num_local_experts,
            target_rank=state.target_rank,
            target_slot=state.target_slot,
            alias_expert=state.target_rank * stride + num_local_experts + state.target_slot,
            moved_tokens=state.moved_tokens,
        )
        for state in replica_states
    )

    flat_logical = selected_experts.reshape(-1)
    flat_logical_long = flat_logical.to(dtype=torch.long)
    flat_physical = (
        torch.div(flat_logical_long, num_local_experts, rounding_mode="floor") * stride
        + torch.remainder(flat_logical_long, num_local_experts)
    ).clone()
    expert_offsets = [0] * num_experts
    for replica, row_allocations in zip(replicas, allocations_by_replica, strict=True):
        local_moved = row_allocations[ep_rank]
        if local_moved == 0:
            continue
        positions = torch.nonzero(flat_logical == replica.source_expert, as_tuple=False).flatten()
        start = expert_offsets[replica.source_expert]
        end = start + local_moved
        if end > positions.numel():
            raise RuntimeError("Concrete local alias rewrite exceeded the selected expert occurrences.")
        flat_physical[positions[start:end]] = replica.alias_expert
        expert_offsets[replica.source_expert] = end
    selected_physical_experts = flat_physical.reshape_as(selected_experts)

    physical_slots_per_rank = stride
    local_physical_counts = torch.zeros((ep_size, physical_slots_per_rank), dtype=torch.long)
    start = ep_rank * num_local_experts
    local_physical_counts[:, :num_local_experts] = remaining_counts[:, start : start + num_local_experts]
    for replica, row_allocations in zip(replicas, allocations_by_replica, strict=True):
        if replica.target_rank == ep_rank:
            local_physical_counts[:, num_local_experts + replica.target_slot] = torch.tensor(
                row_allocations, dtype=torch.long
            )

    input_splits = remaining_counts[ep_rank].reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    for replica, row_allocations in zip(replicas, allocations_by_replica, strict=True):
        input_splits[replica.target_rank] += row_allocations[ep_rank]
    output_splits = tuple(int(value) for value in local_physical_counts.sum(dim=1).tolist())

    if sum(input_splits) != selected_experts.numel():
        raise RuntimeError("The planned input splits do not conserve local selected occurrences.")
    if int(local_physical_counts.sum().item()) != rank_loads_after[ep_rank]:
        raise RuntimeError("The local physical count matrix does not match the planned rank load.")
    for replica, row_allocations in zip(replicas, allocations_by_replica, strict=True):
        if sum(row_allocations) != replica.moved_tokens:
            raise RuntimeError("Replica row allocations do not match moved_tokens metadata.")
        local_alias_count = int((selected_physical_experts == replica.alias_expert).sum().item())
        if local_alias_count != row_allocations[ep_rank]:
            raise RuntimeError("The local alias rewrite does not match its deterministic row allocation.")

    return EPBalancePlan(
        ep_size=ep_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        max_replicas_per_rank=max_replicas_per_rank,
        selected_physical_experts=selected_physical_experts,
        input_splits=tuple(input_splits),
        output_splits=output_splits,
        tokens_per_local_physical_expert=local_physical_counts,
        replicas=replicas,
        rank_loads_before=tuple(rank_loads_before),
        rank_loads_after=tuple(rank_loads_after),
    )


def _gather_counts_with_kernel_metadata(local_counts, ep_group, metadata):
    """Check configuration agreement in the existing histogram collective."""
    local = torch.cat((local_counts, local_counts.new_tensor(metadata)))
    gathered = local.new_empty(dist.get_world_size(ep_group) * local.numel())
    dist.all_gather_into_tensor(gathered, local, group=ep_group)
    gathered = gathered.reshape(-1, local.numel()).cpu()
    configurations = gathered[:, -2:].tolist()
    if any(config != configurations[0] for config in configurations[1:]):
        raise ValueError("Ordered EP weight gradients require identical kernel configurations on every EP rank.")
    return gathered[:, :-2]


def build_ep_balance_plan(
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
    max_replicas_per_rank: int,
) -> EPBalancePlan:
    """Gather the current EP histogram and build this rank's physical alias plan."""
    if not isinstance(selected_experts, torch.Tensor):
        raise TypeError("selected_experts must be a torch.Tensor.")
    if selected_experts.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("selected_experts must contain integer expert IDs.")
    if ep_group is None:
        raise ValueError("build_ep_balance_plan requires a real expert-parallel process group.")
    ep_size = dist.get_world_size(group=ep_group)
    ep_rank = dist.get_rank(group=ep_group)
    if ep_size <= 1:
        raise ValueError("EP load balancing requires ep_size greater than one.")
    if num_experts <= 0 or num_experts % ep_size != 0:
        raise ValueError(f"num_experts ({num_experts}) must be positive and divisible by ep_size ({ep_size}).")
    num_local_experts = num_experts // ep_size
    if max_replicas_per_rank <= 0:
        raise ValueError("max_replicas_per_rank must be positive.")
    if max_replicas_per_rank > num_local_experts:
        raise ValueError(
            "max_replicas_per_rank must be no greater than num_local_experts "
            f"({num_local_experts}), got {max_replicas_per_rank}."
        )
    # The planner already needs host-side integer metadata. Transfer IDs once
    # on accelerators so validation and each concrete replica rewrite do not introduce
    # separate device synchronizations. Preserve the public result device.
    host_planning = selected_experts.device.type in ("cuda", "npu")
    planning_selected = selected_experts.cpu() if host_planning else selected_experts
    flat_selected = planning_selected.reshape(-1)
    if flat_selected.numel() and (
        int(flat_selected.min().item()) < 0 or int(flat_selected.max().item()) >= num_experts
    ):
        raise ValueError(f"selected_experts IDs must be in [0, {num_experts}).")
    local_counts = torch.bincount(flat_selected.to(dtype=torch.long), minlength=num_experts)
    if host_planning:
        local_counts = local_counts.to(selected_experts.device)
    granularity = 1
    if selected_experts.is_cuda:
        from .ep_grad import weight_grad_config_metadata

        metadata = weight_grad_config_metadata()
        granularity = metadata[0]
        gathered_counts = _gather_counts_with_kernel_metadata(local_counts, ep_group, metadata)
    else:
        gathered_counts_flat = local_counts.new_empty(ep_size * num_experts)
        dist.all_gather_into_tensor(gathered_counts_flat, local_counts, group=ep_group)
        gathered_counts = gathered_counts_flat.reshape(ep_size, num_experts)
    plan = _build_ep_balance_plan_from_counts(
        selected_experts=planning_selected,
        tokens_per_expert_by_rank=gathered_counts,
        num_experts=num_experts,
        ep_rank=ep_rank,
        max_replicas_per_rank=max_replicas_per_rank,
        token_granularity=granularity,
    )
    if host_planning:
        plan = replace(plan, selected_physical_experts=plan.selected_physical_experts.to(selected_experts.device))
    return plan


@dataclass(frozen=True)
class _Qwen3_5MoEEPLoadBalancer:
    ep_group: dist.ProcessGroup
    num_experts: int
    max_replicas_per_rank: int
    layer_index: int
    buffer_pool: ReplicaBufferPool | None = None

    @property
    def optimize_memory(self) -> bool:
        return self.buffer_pool is not None

    def build_plan(self, selected_experts: torch.Tensor) -> EPBalancePlan:
        return build_ep_balance_plan(
            selected_experts=selected_experts,
            num_experts=self.num_experts,
            ep_group=self.ep_group,
            max_replicas_per_rank=self.max_replicas_per_rank,
        )


def _is_runtime_qwen3_5_merged_experts(module: nn.Module) -> bool:
    class_name = module.__class__.__name__
    if class_name != "Qwen3_5MoeExperts" and not (
        class_name.startswith("FSDP") and class_name.endswith("Qwen3_5MoeExperts")
    ):
        return False
    gate_up_proj = getattr(module, "gate_up_proj", None)
    down_proj = getattr(module, "down_proj", None)
    return getattr(gate_up_proj, "ndim", None) == 3 and getattr(down_proj, "ndim", None) == 3


def attach_qwen3_5_moe_ep_load_balancers(
    model: nn.Module, max_replicas_per_rank: int, *, optimize_memory: bool = False
) -> int:
    """Attach a private planner controller to each compatible Qwen3.5 MoE layer."""
    parallel_state = get_parallel_state()
    ep_group = parallel_state.ep_group
    if ep_group is None:
        raise ValueError("Qwen3.5 MoE EP load balancing requires an initialized EP process group.")
    ep_size = dist.get_world_size(group=ep_group)
    if ep_size <= 1:
        raise ValueError("Qwen3.5 MoE EP load balancing requires ep_size greater than one.")
    if max_replicas_per_rank <= 0:
        raise ValueError("max_replicas_per_rank must be positive.")

    expert_modules = [module for _, module in model.named_modules() if _is_runtime_qwen3_5_merged_experts(module)]
    validated_num_experts = []
    for module in expert_modules:
        num_experts = getattr(module, "num_experts", None)
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("Qwen3_5MoeExperts.num_experts must be a positive integer.")
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")
        num_local_experts = num_experts // ep_size
        if max_replicas_per_rank > num_local_experts:
            raise ValueError(
                "max_replicas_per_rank must be no greater than num_local_experts "
                f"({num_local_experts}), got {max_replicas_per_rank}."
            )
        gate_up_rows = module.gate_up_proj.shape[0]
        down_rows = module.down_proj.shape[0]
        if gate_up_rows != num_local_experts or down_rows != num_local_experts:
            if gate_up_rows == num_experts and down_rows == num_experts:
                raise ValueError(
                    "Merged Qwen3_5MoeExperts weights still have the global/pre-FSDP row count "
                    f"({num_experts}); attach EP load balancers only after ParallelPlan EP sharding and FSDP2 "
                    f"wrapping, when both weights have {num_local_experts} EP-local expert rows."
                )
            raise ValueError(
                "Merged Qwen3_5MoeExperts weights must each have exactly "
                f"{num_local_experts} EP-local expert rows after ParallelPlan/FSDP2 wrapping; "
                f"got gate_up_proj={gate_up_rows} and down_proj={down_rows}."
            )
        validated_num_experts.append(num_experts)

    for layer_index, (module, num_experts) in enumerate(zip(expert_modules, validated_num_experts, strict=True)):
        module._veomni_ep_load_balancer = _Qwen3_5MoEEPLoadBalancer(
            ep_group=ep_group,
            num_experts=num_experts,
            max_replicas_per_rank=max_replicas_per_rank,
            layer_index=layer_index,
            buffer_pool=ReplicaBufferPool() if optimize_memory else None,
        )
    return len(expert_modules)
