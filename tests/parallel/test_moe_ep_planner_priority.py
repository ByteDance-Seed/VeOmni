"""Keep the allocation policy identical to the original full-scan planner."""

import itertools

import pytest
import torch

from tests.parallel.test_moe_ep_load_balance_planner import ep_load_balance as ep


def _reference_plan_replicas(
    counts: torch.Tensor,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    max_replicas_per_rank: int,
) -> tuple[list[ep._ReplicaState], list[int], list[int], list[dict[int, int]]]:
    expert_totals = [int(value) for value in counts.sum(dim=0).tolist()]
    rank_loads_before = [
        sum(expert_totals[rank * num_local_experts : (rank + 1) * num_local_experts]) for rank in range(ep_size)
    ]
    rank_loads = rank_loads_before.copy()
    expert_allocations = [{expert // num_local_experts: total} for expert, total in enumerate(expert_totals)]
    replica_states: list[ep._ReplicaState] = []
    replica_indices_by_expert: list[list[int]] = [[] for _ in range(num_experts)]
    used_slots: list[set[int]] = [set() for _ in range(ep_size)]

    while True:
        # Integer loads that differ by at most one already attain the minimum
        # descending-sorted profile for this fixed token total. No replica can
        # strictly improve the objective, so avoid searching equivalent moves.
        if max(rank_loads) - min(rank_loads) <= 1:
            break
        candidate_experts = []
        for expert in range(num_experts):
            owner = expert // num_local_experts
            owner_tokens = expert_allocations[expert].get(owner, 0)
            existing_targets = {replica_states[index].target_rank for index in replica_indices_by_expert[expert]}
            has_target = any(
                rank != owner and rank not in existing_targets and len(used_slots[rank]) < max_replicas_per_rank
                for rank in range(ep_size)
            )
            if owner_tokens > 0 and has_target:
                candidate_experts.append(expert)

        candidate_experts.sort(
            key=lambda expert: (
                -rank_loads[expert // num_local_experts],
                -expert_allocations[expert].get(expert // num_local_experts, 0),
                expert // num_local_experts,
                expert,
            )
        )

        accepted = None
        current_profile = ep._rank_load_profile(rank_loads)
        for expert in candidate_experts:
            owner = expert // num_local_experts
            existing_indices = replica_indices_by_expert[expert]
            existing_targets = {replica_states[index].target_rank for index in existing_indices}
            target_candidates = [
                rank
                for rank in range(ep_size)
                if rank != owner and rank not in existing_targets and len(used_slots[rank]) < max_replicas_per_rank
            ]
            target_candidates.sort(key=lambda rank: (rank_loads[rank], rank))

            for target in target_candidates:
                locations = [owner, *(replica_states[index].target_rank for index in existing_indices), target]
                base_loads = {rank: rank_loads[rank] - expert_allocations[expert].get(rank, 0) for rank in locations}
                allocations = ep._waterfill_allocations(expert_totals[expert], base_loads)
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
                if ep._rank_load_profile(proposed_loads) >= current_profile:
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
        replica_states.append(
            ep._ReplicaState(
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


@pytest.mark.parametrize("ep_size", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("local_experts", [1, 3, 32])
@pytest.mark.parametrize("slots", [1, 2])
def test_heap_planner_matches_original_full_priority_scan(ep_size, local_experts, slots):
    if slots > local_experts:
        pytest.skip("Replica slots cannot exceed local experts")
    experts = ep_size * local_experts
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        counts = torch.randint(0, 32, (ep_size, experts), generator=generator)
        if seed % 4 == 0:
            counts[:, 0] += 1024
        elif seed % 4 == 1:
            counts[0].zero_()
        elif seed % 4 == 2:
            counts.fill_(seed)
        args = (counts, ep_size, experts, local_experts, slots)
        assert ep._plan_replicas(*args) == _reference_plan_replicas(*args), seed


def test_target_masks_support_more_than_sixty_four_ranks():
    counts = torch.zeros(65, 65, dtype=torch.long)
    counts[:, 0] = 1
    args = (counts, 65, 65, 1, 1)
    assert ep._plan_replicas(*args) == _reference_plan_replicas(*args)


@pytest.mark.parametrize("slots", [1, 2])
def test_first_replica_pruning_exhaustive_integer_ties(slots):
    for totals in itertools.product(range(5), repeat=4):
        counts = torch.tensor([totals, (0, 0, 0, 0)], dtype=torch.long)
        args = (counts, 2, 4, 2, slots)
        assert ep._plan_replicas(*args) == _reference_plan_replicas(*args), totals
