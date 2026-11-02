import gc
import importlib
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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


def _transposed_gradient_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        plan = _plan(ep_size=2, replicas=(_replica(source_rank=0, target_rank=1),))
        for rows, columns in ((2, 2), (4, 2), (2, 3)):
            local_weight = nn.Parameter(torch.zeros(1, rows, columns))
            replica_weight = ep_load_balance.ExpertReplicaTransfer.start(local_weight, plan, dist.group.WORLD).wait()
            combined = ep_load_balance.cat_local_and_replica_weights(
                local_weight, replica_weight, plan, dist.group.WORLD
            )

            # NPU merged GEMM transposes expert weights before computing them.
            base = torch.arange(2 * rows * columns, dtype=torch.float32).reshape(2, columns, rows).transpose(1, 2)
            gradient = base + 100 * rank
            assert not gradient.is_contiguous()
            combined.backward(gradient)
            expected = (base[0] + base[1] + 100) if rank == 0 else base[0] + 100
            torch.testing.assert_close(local_weight.grad, expected.unsqueeze(0), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_transposed_replica_gradients_use_contiguous_receive_buffers(tmp_path: Path):
    rendezvous = f"file://{tmp_path / 'executor-transposed-gradient'}"
    mp.spawn(_transposed_gradient_worker, args=(2, rendezvous), nprocs=2, join=True)


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


def _buffer_pool(max_cached=2):
    return ep_load_balance.ReplicaBufferPool(max_cached=max_cached)


def test_pool_refreshes_reused_storage_without_reusing_live_leases():
    pool = _buffer_pool()
    reference = torch.empty(2)
    first = pool.acquire(reference, (2, 3))
    first_tensor = first.tensor
    first_tensor.fill_(7)
    live = pool.acquire(reference, (2, 3))
    assert live.tensor.data_ptr() != first_tensor.data_ptr()
    first.record_use()
    first.release()
    first.release()
    with pytest.raises(RuntimeError, match="released"):
        first.record_use()
    with pytest.raises(RuntimeError, match="released"):
        _ = first.tensor
    recycled = pool.acquire(reference, (2, 3))
    assert recycled.tensor.data_ptr() == first_tensor.data_ptr()
    assert torch.equal(recycled.tensor, torch.zeros(2, 3))
    recycled.tensor.fill_(11)
    assert torch.equal(live.tensor, torch.zeros(2, 3))


def test_pool_separates_shape_dtype_and_bounds_total_idle_storage():
    pool = _buffer_pool()
    leases = [
        pool.acquire(torch.empty(1, dtype=dtype), shape)
        for dtype, shape in ((torch.float32, (2,)), (torch.float64, (2,)), (torch.float32, (3,)))
    ]
    tensors = [lease.tensor for lease in leases]
    for lease in leases:
        lease.record_use()
        lease.release()
    borrowed = [pool.acquire(torch.empty_like(tensor), tensor.shape) for tensor in tensors]
    reused = sum(lease.tensor.data_ptr() == tensor.data_ptr() for lease, tensor in zip(borrowed, tensors))
    assert reused == 2
    for lease, tensor in zip(borrowed, tensors):
        assert lease.tensor.shape == tensor.shape
        assert lease.tensor.dtype == tensor.dtype


def test_concurrent_returns_preserve_total_idle_cap():
    pool = _buffer_pool()
    leases = [pool.acquire(torch.empty(1), (4,)) for _ in range(8)]
    tensors = [lease.tensor for lease in leases]
    for lease in leases:
        lease.record_use()

    class YieldingList(list):
        def __len__(self):
            size = super().__len__()
            # Force a scheduling point between the size check and append.
            # The stored tensors and actual cache operations remain real.
            time.sleep(0.01)
            return size

    pool._idle = YieldingList(pool._idle)
    barrier = threading.Barrier(len(leases))

    def release(lease):
        barrier.wait()
        lease.release()

    with ThreadPoolExecutor(max_workers=len(leases)) as executor:
        list(executor.map(release, leases))
    borrowed = [pool.acquire(tensors[0], (4,)) for _ in leases]
    old_pointers = {tensor.data_ptr() for tensor in tensors}
    assert sum(lease.tensor.data_ptr() in old_pointers for lease in borrowed) == 2
    assert len({lease.tensor.data_ptr() for lease in borrowed}) == len(borrowed)


def test_unrecorded_lease_is_not_cached():
    pool = _buffer_pool()
    lease = pool.acquire(torch.empty(1), (4,))
    old_tensor = lease.tensor
    lease.release()
    assert pool.acquire(old_tensor, old_tensor.shape).tensor.data_ptr() != old_tensor.data_ptr()


@pytest.mark.parametrize("failure", [False, True])
def test_unfinished_or_failed_transfer_cannot_reenter_pool(failure):
    pool = _buffer_pool()
    lease = pool.acquire(torch.empty(1), (4,))
    old_tensor = lease.tensor

    class FailingWork:
        def wait(self):
            raise RuntimeError("transfer failed")

    transfer = ep_load_balance.ExpertReplicaTransfer(old_tensor, (FailingWork(),), (), lease=lease)
    with pytest.raises(RuntimeError, match="wait"):
        transfer.record_stream_use()
    if failure:
        with pytest.raises(RuntimeError, match="transfer failed"):
            transfer.wait()
    del transfer, lease
    gc.collect()
    assert pool.acquire(old_tensor, old_tensor.shape).tensor.data_ptr() != old_tensor.data_ptr()


class _GraphHeldReplica(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, transfer):
        ctx.transfer = transfer
        result = value * transfer.wait()
        transfer.record_stream_use()
        return result

    @staticmethod
    def backward(ctx, gradient):
        result = gradient * ctx.transfer.wait()
        ctx.transfer.record_stream_use()
        return result, None


def test_graph_owned_transfer_survives_retained_backward_and_refreshes_after_gc():
    pool = _buffer_pool()
    value = torch.ones(4, requires_grad=True)
    lease = pool.acquire(value, value.shape)
    old_tensor = lease.tensor
    old_tensor.fill_(3)
    transfer = ep_load_balance.ExpertReplicaTransfer(old_tensor, (), (), lease=lease)
    output = _GraphHeldReplica.apply(value, transfer)
    del transfer, lease
    output.sum().backward(retain_graph=True)
    assert torch.equal(value.grad, torch.full((4,), 3.0))
    other = pool.acquire(value, value.shape)
    other.tensor.fill_(19)
    assert other.tensor.data_ptr() != old_tensor.data_ptr()
    value.grad = None
    output.sum().backward(retain_graph=True)
    assert torch.equal(value.grad, torch.full((4,), 3.0))
    del output
    gc.collect()
    recycled = pool.acquire(value, value.shape)
    assert recycled.tensor.data_ptr() == old_tensor.data_ptr()
    assert torch.equal(recycled.tensor, torch.zeros(4))


def test_no_grad_transfer_recycles_on_gc_without_graph():
    pool = _buffer_pool()
    value = torch.ones(4)
    lease = pool.acquire(value, value.shape)
    old_tensor = lease.tensor
    old_tensor.fill_(5)
    transfer = ep_load_balance.ExpertReplicaTransfer(old_tensor, (), (), lease=lease)
    with torch.no_grad():
        result = _GraphHeldReplica.apply(value, transfer)
    del transfer, lease
    gc.collect()
    recycled = pool.acquire(value, value.shape)
    assert recycled.tensor.data_ptr() == old_tensor.data_ptr()
    assert torch.equal(result, torch.full((4,), 5.0))


def _pooled_gradient_parts_worker(rank, world_size, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        pool = _buffer_pool()
        plan = _plan(ep_size=2, replicas=(_replica(source_rank=0, target_rank=1),))
        previous = None
        for step in range(2):
            weight = torch.full((1, 2, 3), float(10 + rank + step))
            transfer = ep_load_balance.ExpertReplicaTransfer.start(weight, plan, dist.group.WORLD, pool=pool)
            replica = transfer.wait()
            if previous is not None:
                assert replica.data_ptr() == previous.data_ptr()
            assert torch.equal(replica, torch.full_like(replica, float(10 + step) if rank else 0.0))
            local_grad = torch.arange(6.0).reshape(1, 3, 2).transpose(1, 2) + rank * 10
            replica_grad = torch.full((1, 3, 2), float(20 + rank)).transpose(1, 2)
            original = local_grad.clone()
            result = ep_load_balance._merge_replica_weight_gradient_parts(
                local_grad, replica_grad, plan, dist.group.WORLD
            )
            assert torch.equal(result, original + 21 if rank == 0 else original)
            assert torch.equal(local_grad, original)
            assert result.dtype == local_grad.dtype
            transfer.record_stream_use()
            previous = replica
            del transfer
            gc.collect()
    finally:
        dist.destroy_process_group()


def test_pooled_two_rank_refresh_and_noncontiguous_gradient_parts(tmp_path):
    mp.spawn(_pooled_gradient_parts_worker, args=(2, f"file://{tmp_path / 'pooled'}"), nprocs=2, join=True)


@pytest.mark.parametrize("failure_stage", ["record_stream", "event"])
def test_stream_record_failure_permanently_excludes_storage(monkeypatch, failure_stage):
    # The host has no accelerator. Substitute only the stream API boundary;
    # the lease, pool, tensor storage and pointer-reuse checks are real.
    buffer_module = sys.modules[ep_load_balance.ReplicaBufferPool.__module__]
    failing = True

    class StreamTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("meta")

        def record_stream(self, stream):
            if failing and failure_stage == "record_stream":
                raise RuntimeError("stream tracking failed")

    class Event:
        def record(self, stream):
            if failing and failure_stage == "event":
                raise RuntimeError("stream tracking failed")

    api = types.SimpleNamespace(device=lambda device: nullcontext(), current_stream=lambda: object())
    monkeypatch.setattr(buffer_module, "_device_api", lambda tensor: api)
    monkeypatch.setattr(buffer_module, "create_event", Event)
    pool = _buffer_pool()
    tensor = torch.ones(4).as_subclass(StreamTensor)
    lease = buffer_module.ReplicaBufferLease(pool, tensor)
    with pytest.raises(RuntimeError, match="tracking failed"):
        lease.record_use()
    failing = False
    lease.record_use()
    lease.release()
    replacement = pool.acquire(tensor, tensor.shape)
    assert replacement.tensor.data_ptr() != tensor.data_ptr()
    assert torch.equal(tensor, torch.ones(4))


def test_inference_forward_storage_can_be_reused_by_training():
    pool = _buffer_pool()
    with torch.inference_mode():
        lease = pool.acquire(torch.ones(4), (4,))
        pointer = lease.tensor.data_ptr()
        lease.record_use()
        lease.release()
    lease = pool.acquire(torch.ones(4), (4,))
    assert lease.tensor.data_ptr() == pointer
    lease.tensor.fill_(3)
    value = torch.ones(4, requires_grad=True)
    (value * lease.tensor).sum().backward()
    assert torch.equal(value.grad, torch.full((4,), 3.0))


def test_accelerator_pool_waits_for_every_stream_before_reuse():
    from veomni.utils.device import (
        IS_CUDA_AVAILABLE,
        IS_NPU_AVAILABLE,
        create_stream,
        get_device_type,
        get_torch_device,
    )

    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        pytest.skip("Requires CUDA or NPU stream execution.")
    api = get_torch_device()
    reference = torch.empty(512 * 512, device=get_device_type())
    pool = _buffer_pool()
    streams = [create_stream() for _ in range(3)]
    previous = api.current_stream()
    for stream in streams:
        stream.wait_stream(previous)
    lease = pool.acquire(reference, reference.shape)
    pointer = lease.tensor.data_ptr()
    # Independent consumers on two streams: waiting for just the last event
    # cannot protect the first stream's read from the third stream's zero_.
    lease.tensor.fill_(7)
    for stream in streams[:2]:
        stream.wait_stream(api.current_stream())
    results = []
    for index, stream in enumerate(streams[:2]):
        with api.stream(stream):
            delay = torch.ones((512, 512), device=reference.device)
            for _ in range(16 if index == 0 else 2):
                delay = delay @ delay
            results.append(lease.tensor.clone())
            lease.record_use()
    lease.release()
    with api.stream(streams[2]):
        replacement = pool.acquire(reference, reference.shape)
        assert replacement.tensor.data_ptr() == pointer
        zeroed = replacement.tensor.clone()
        replacement.tensor.fill_(19)
        replacement.record_use()
        replacement.release()
    for stream in streams:
        stream.synchronize()
    assert torch.equal(zeroed, torch.zeros_like(reference))
    for result in results:
        assert torch.equal(result, torch.full_like(reference, 7))


def test_accelerator_eviction_preserves_allocator_stream_ownership():
    from veomni.utils.device import (
        IS_CUDA_AVAILABLE,
        IS_NPU_AVAILABLE,
        create_stream,
        get_device_type,
        get_torch_device,
    )

    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        pytest.skip("Requires CUDA or NPU allocator stream execution.")
    api = get_torch_device()
    reference = torch.empty(512 * 512, device=get_device_type())
    pool = _buffer_pool(max_cached=0)
    allocation_stream, consumer_stream = create_stream(), create_stream()
    with api.stream(allocation_stream):
        lease = pool.acquire(reference, reference.shape)
        lease.tensor.fill_(7)
    consumer_stream.wait_stream(allocation_stream)
    with api.stream(consumer_stream):
        delay = torch.ones((512, 512), device=reference.device)
        for _ in range(16):
            delay = delay @ delay
        result = lease.tensor.clone()
        lease.record_use()
    lease.release()
    with api.stream(allocation_stream):
        # Repeated same-sized allocations pressure the freed storage's bin.
        for _ in range(16):
            torch.empty_like(reference).fill_(19)
    allocation_stream.synchronize()
    consumer_stream.synchronize()
    assert torch.equal(result, torch.full_like(reference, 7))
