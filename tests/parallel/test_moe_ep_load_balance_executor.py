import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


# The existing MoE package initializer imports Triton kernels eagerly. Load this
# pure CPU executor submodule without that unrelated optional dependency.
_MOE_PACKAGE_NAME = "veomni.distributed.moe"
_previous_moe_package = sys.modules.get(_MOE_PACKAGE_NAME)
_moe_package = types.ModuleType(_MOE_PACKAGE_NAME)
_moe_package.__path__ = [str(Path(__file__).parents[2] / "veomni" / "distributed" / "moe")]
sys.modules[_MOE_PACKAGE_NAME] = _moe_package
ep_load_balance = importlib.import_module(f"{_MOE_PACKAGE_NAME}.ep_load_balance")
if _previous_moe_package is None:
    del sys.modules[_MOE_PACKAGE_NAME]
else:
    sys.modules[_MOE_PACKAGE_NAME] = _previous_moe_package


def test_executor_symbols_are_publicly_exported():
    assert "ExpertReplicaTransfer" in ep_load_balance.__all__
    assert "cat_local_and_replica_weights" in ep_load_balance.__all__


def _plan(
    *,
    ep_size: int,
    replicas: tuple[ep_load_balance.ExpertReplica, ...],
    max_replicas_per_rank: int = 1,
) -> ep_load_balance.EPBalancePlan:
    return ep_load_balance.EPBalancePlan(
        ep_size=ep_size,
        num_experts=ep_size,
        num_local_experts=1,
        max_replicas_per_rank=max_replicas_per_rank,
        selected_physical_experts=torch.empty(0, dtype=torch.long),
        input_splits=(0,) * ep_size,
        output_splits=(0,) * ep_size,
        tokens_per_local_physical_expert=torch.zeros((ep_size, 1 + max_replicas_per_rank), dtype=torch.long),
        replicas=replicas,
        rank_loads_before=(0,) * ep_size,
        rank_loads_after=(0,) * ep_size,
    )


def _replica(
    *,
    source_rank: int,
    target_rank: int,
    target_slot: int = 0,
    moved_tokens: int = 1,
) -> ep_load_balance.ExpertReplica:
    stride = 2
    return ep_load_balance.ExpertReplica(
        source_expert=source_rank,
        source_rank=source_rank,
        source_local_expert=0,
        target_rank=target_rank,
        target_slot=target_slot,
        alias_expert=target_rank * stride + 1 + target_slot,
        moved_tokens=moved_tokens,
    )


class _LocalExperts(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(value)


def _exercise_two_rank_copy_and_gradient(rank: int, group: dist.ProcessGroup) -> None:
    selected_experts = torch.zeros((6 if rank == 0 else 4, 1), dtype=torch.long)
    plan = ep_load_balance.build_ep_balance_plan(selected_experts, 2, group, max_replicas_per_rank=1)
    assert plan.replicas == (_replica(source_rank=0, target_rank=1, moved_tokens=5),)
    experts = _LocalExperts(torch.tensor([[2.0, 3.0]]) if rank == 0 else torch.tensor([[5.0, 7.0]]))

    transfer = ep_load_balance.ExpertReplicaTransfer.start(experts.weight, plan, group)
    replica_weight = transfer.wait()

    assert transfer.wait() is replica_weight
    assert replica_weight.shape == (1, 2)
    assert replica_weight.dtype == experts.weight.dtype
    assert replica_weight.device == experts.weight.device
    assert torch.equal(replica_weight, torch.tensor([[0.0, 0.0]]) if rank == 0 else torch.tensor([[2.0, 3.0]]))
    assert not isinstance(replica_weight, nn.Parameter)
    assert [name for name, _ in experts.named_parameters()] == ["weight"]
    assert list(experts.state_dict()) == ["weight"]

    physical_weight = ep_load_balance.cat_local_and_replica_weights(experts.weight, replica_weight, plan, group)
    if rank == 1:
        alias_result = physical_weight[1].matmul(torch.tensor([10.0, 1.0]))
        assert alias_result.item() == 23.0
        loss = physical_weight[0].sum() * 2 + physical_weight[1].mul(torch.tensor([3.0, 4.0])).sum()
    else:
        loss = physical_weight[0].sum()
    loss.backward()

    expected_grad = torch.tensor([[4.0, 5.0]]) if rank == 0 else torch.tensor([[2.0, 2.0]])
    assert torch.equal(experts.weight.grad, expected_grad)

    dist.barrier(group=group)
    bidirectional_plan = _plan(
        ep_size=2,
        replicas=(
            _replica(source_rank=0, target_rank=1),
            _replica(source_rank=1, target_rank=0),
        ),
    )
    bidirectional_weight = nn.Parameter(torch.tensor([[7.0 + rank]]))
    bidirectional_replica = ep_load_balance.ExpertReplicaTransfer.start(
        bidirectional_weight, bidirectional_plan, group
    ).wait()
    assert bidirectional_replica.item() == 8.0 - rank
    bidirectional_combined = ep_load_balance.cat_local_and_replica_weights(
        bidirectional_weight, bidirectional_replica, bidirectional_plan, group
    )
    (bidirectional_combined[0] + bidirectional_combined[1] * (rank + 2)).backward()
    assert bidirectional_weight.grad.item() == 4.0 - rank


def _two_rank_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        _exercise_two_rank_copy_and_gradient(rank, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_two_rank_copy_alias_computation_and_replica_gradient_return(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'executor-two-rank'}"
    mp.spawn(_two_rank_worker, args=(2, rendezvous), nprocs=2, join=True)


def test_torchrun_two_rank_copy_alias_computation_and_replica_gradient_return():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run this test with torchrun --standalone --nproc-per-node=2.")
    dist.init_process_group("gloo")
    try:
        _exercise_two_rank_copy_and_gradient(dist.get_rank(), dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def _empty_plan_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        plan = _plan(ep_size=world_size, replicas=())
        local_weight = nn.Parameter(torch.tensor([[float(rank + 1)]]))
        with pytest.raises(ValueError, match="expert rows"):
            ep_load_balance.ExpertReplicaTransfer.start(torch.zeros((2, 1)), plan, dist.group.WORLD)
        with pytest.raises(ValueError, match="process-group size"):
            ep_load_balance.ExpertReplicaTransfer.start(local_weight, _plan(ep_size=3, replicas=()), dist.group.WORLD)
        with pytest.raises(ValueError, match="shape"):
            ep_load_balance.cat_local_and_replica_weights(local_weight, torch.zeros((2, 1)), plan, dist.group.WORLD)
        with pytest.raises(TypeError, match="same dtype"):
            ep_load_balance.cat_local_and_replica_weights(
                local_weight, torch.zeros((1, 1), dtype=torch.double), plan, dist.group.WORLD
            )
        with pytest.raises(TypeError, match="not an nn.Parameter"):
            ep_load_balance.cat_local_and_replica_weights(
                local_weight, nn.Parameter(torch.zeros((1, 1))), plan, dist.group.WORLD
            )
        original_isend = ep_load_balance.dist.isend
        original_irecv = ep_load_balance.dist.irecv

        def unexpected_p2p(*args, **kwargs):
            raise AssertionError("An empty plan must not issue point-to-point communication.")

        ep_load_balance.dist.isend = unexpected_p2p
        ep_load_balance.dist.irecv = unexpected_p2p
        try:
            replica_weight = ep_load_balance.ExpertReplicaTransfer.start(local_weight, plan, dist.group.WORLD).wait()
            combined = ep_load_balance.cat_local_and_replica_weights(
                local_weight, replica_weight, plan, dist.group.WORLD
            )
        finally:
            ep_load_balance.dist.isend = original_isend
            ep_load_balance.dist.irecv = original_irecv

        assert replica_weight.shape == (1, 1)
        assert torch.count_nonzero(replica_weight).item() == 0
        assert combined is local_weight
        combined.sum().backward()
        assert torch.equal(local_weight.grad, torch.ones_like(local_weight))
    finally:
        dist.destroy_process_group()


def test_empty_plan_is_an_identity_without_p2p(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'executor-empty'}"
    mp.spawn(_empty_plan_worker, args=(2, rendezvous), nprocs=2, join=True)


def _multiple_copy_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        replicas = tuple(_replica(source_rank=0, target_rank=target) for target in range(1, 4))
        plan = _plan(ep_size=world_size, replicas=replicas)
        local_weight = nn.Parameter(torch.tensor([[10.0 + rank]]))
        replica_weight = ep_load_balance.ExpertReplicaTransfer.start(local_weight, plan, dist.group.WORLD).wait()

        expected_replica = 10.0 if rank else 0.0
        assert replica_weight.item() == expected_replica
        combined = ep_load_balance.cat_local_and_replica_weights(local_weight, replica_weight, plan, dist.group.WORLD)
        loss = combined[0].sum() if rank == 0 else combined[1].sum() * (rank + 1)
        loss.backward()

        expected_grad = 10.0 if rank == 0 else 0.0
        assert local_weight.grad.item() == expected_grad
    finally:
        dist.destroy_process_group()


def test_multiple_copies_of_one_source_sum_all_replica_gradients(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'executor-multiple-copy'}"
    mp.spawn(_multiple_copy_worker, args=(4, rendezvous), nprocs=4, join=True)


def _noncontiguous_group_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        even_group = dist.new_group((0, 2), backend="gloo")
        odd_group = dist.new_group((1, 3), backend="gloo")
        ep_group = even_group if rank % 2 == 0 else odd_group
        ep_rank = dist.get_rank(ep_group)
        plan = _plan(ep_size=2, replicas=(_replica(source_rank=0, target_rank=1),))
        local_weight = nn.Parameter(torch.tensor([[100.0 + rank]]))

        replica_weight = ep_load_balance.ExpertReplicaTransfer.start(local_weight, plan, ep_group).wait()
        expected_replica = 0.0 if ep_rank == 0 else 100.0 + (rank - 2)
        assert replica_weight.item() == expected_replica

        combined = ep_load_balance.cat_local_and_replica_weights(local_weight, replica_weight, plan, ep_group)
        loss = combined[0].sum() if ep_rank == 0 else combined[1].sum() * (rank + 1)
        loss.backward()
        expected_grad = float(rank + 4) if ep_rank == 0 else 0.0
        assert local_weight.grad.item() == expected_grad
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_noncontiguous_process_groups_map_ep_ranks_to_global_peers(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'executor-noncontiguous'}"
    mp.spawn(_noncontiguous_group_worker, args=(4, rendezvous), nprocs=4, join=True)
