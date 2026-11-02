"""Regression for rounding across the expert/replica autograd boundary."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, train_gate, train_down):
    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
    from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        plan = ep.build_ep_balance_plan(torch.tensor([[0]]), 2, dist.group.WORLD, 1)
        assert len(plan.replicas) == 1
        assert plan.replicas[0].source_rank == 0 and plan.replicas[0].target_rank == 1
        gate = torch.zeros(1, 2, 3, dtype=torch.bfloat16, requires_grad=train_gate)
        down = torch.zeros(1, 3, 1, dtype=torch.bfloat16, requires_grad=train_down)
        tokens = torch.ones(1, 3, dtype=torch.bfloat16, requires_grad=True)

        def forward(ctx, tokens, cumsum, combined_gate, combined_down, limit):
            ctx.save_for_backward(combined_gate, combined_down)
            return tokens.clone()

        def backward(ctx, grad, *, weight_gradient_dtype=None, weight_needs_grad=None):
            assert weight_gradient_dtype == torch.float32
            outputs = []
            for weight, needed in zip(ctx.saved_tensors, weight_needs_grad, strict=True):
                if not needed:
                    outputs.append(None)
                    continue
                value = torch.zeros_like(weight, dtype=weight_gradient_dtype)
                value[0 if rank == 0 else 1].fill_(257.0 if rank == 0 else -256.0)
                outputs.append(value)
            return grad, None, *outputs, None

        original_forward, original_backward = moe.EPMergedFc1GroupGemm.forward, moe.EPMergedFc1GroupGemm.backward
        moe.EPMergedFc1GroupGemm.forward = staticmethod(forward)
        moe.EPMergedFc1GroupGemm.backward = staticmethod(backward)
        try:
            output = moe._BalancedEPMergedFc1GroupGemm.apply(
                tokens,
                torch.tensor([1, 2]),
                gate,
                down,
                torch.zeros_like(gate),
                torch.zeros_like(down),
                plan,
                dist.group.WORLD,
                None,
            )
            output.sum().backward()
        finally:
            moe.EPMergedFc1GroupGemm.forward = staticmethod(original_forward)
            moe.EPMergedFc1GroupGemm.backward = staticmethod(original_backward)
        for weight, needed in ((gate, train_gate), (down, train_down)):
            if needed:
                assert torch.equal(weight.grad, torch.full_like(weight, 1 if rank == 0 else 0))
                assert weight.grad.dtype == torch.bfloat16
            else:
                assert weight.grad is None
        assert torch.equal(tokens.grad, torch.ones_like(tokens))

        # A separate BF16 Cat edge quantizes 257 to 256 before replica
        # reduction, producing zero rather than the true one.
        old_weight = torch.zeros(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        combined = ep.cat_local_and_replica_weights(old_weight, torch.zeros_like(old_weight), plan, dist.group.WORLD)
        partial = torch.zeros_like(combined, dtype=torch.float32)
        partial[0 if rank == 0 else 1].fill_(257 if rank == 0 else -256)
        (combined.float() * partial).sum().backward()
        assert torch.count_nonzero(old_weight.grad) == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("train_gate,train_down", [(True, True), (True, False), (False, True)])
def test_fp32_replica_gradients_cross_autograd_boundary_before_final_rounding(tmp_path, train_gate, train_down):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'fp32_grad'}", train_gate, train_down), nprocs=2, join=True)


def _reference_nk(a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False, *, b_extra=None):
    """CPU replacement only for the unavailable grouped GEMM, not MLP/autograd."""
    assert not transpose_a
    outputs = []
    start = 0
    for index, end in enumerate(cumsum_M.tolist()):
        weight = b[index] if index < len(b) else b_extra[index - len(b)]
        outputs.append(a[start:end] @ (weight.T if transpose_b else weight))
        start = end
    return torch.cat(outputs, dim=0)


def _reference_mn(a, b, c, cumsum_K, max_K, transpose_a, transpose_b):
    assert transpose_a and not transpose_b
    start = 0
    for index, end in enumerate(cumsum_K.tolist()):
        c[index].copy_(a[start:end].float().T @ b[start:end].float())
        start = end
    return c


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("memory", [False, True])
def test_balanced_gemm_uses_sender_summed_group_bound_in_forward_and_dx(monkeypatch, rank, memory):
    from types import SimpleNamespace

    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
    from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe

    counts = torch.tensor([[140, 70, 10, 10], [140, 70, 10, 10]])
    selected = torch.repeat_interleave(torch.arange(4), counts[rank]).reshape(-1, 1)
    plan = ep._build_ep_balance_plan_from_counts(selected, counts, 4, rank, 1)
    assert plan.replicas
    groups = plan.tokens_per_local_physical_expert.sum(0)
    bound = int(groups.max())
    assert bound > int(plan.tokens_per_local_physical_expert.max())
    assert bound < int(groups.sum())
    monkeypatch.setattr(dist, "get_rank", lambda *args, **kwargs: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda *args, **kwargs: 2)
    monkeypatch.setattr(moe, "group_gemm_same_nk", _reference_nk)
    monkeypatch.setattr(moe, "group_gemm_same_mn", _reference_mn)
    torch.manual_seed(71 + rank)
    tokens = torch.randn(int(groups.sum()), 3, requires_grad=True)
    gate, down = torch.randn(2, 4, 3), torch.randn(2, 3, 2)
    replica_gate, replica_down = torch.randn(1, 4, 3), torch.randn(1, 3, 2)
    cumsum = groups.cumsum(0)
    reference = moe.EPMergedFc1GroupGemm.apply(
        tokens, cumsum, torch.cat((gate, replica_gate)), torch.cat((down, replica_down)), None
    )
    reference.sum().backward()
    expected_dx = tokens.grad.clone()
    tokens.grad = None
    observed = []

    def bounded_gemm(*args, **kwargs):
        observed.append(kwargs["max_M"])
        assert kwargs["max_M"] == bound, "balanced execution launched the total-token grid"
        return _reference_nk(*args, **kwargs)

    monkeypatch.setattr(moe, "group_gemm_same_nk", bounded_gemm)
    args = (tokens, cumsum, gate, down, replica_gate, replica_down, plan, object(), None)
    if memory:
        transfers = (SimpleNamespace(record_stream_use=lambda: None), SimpleNamespace(record_stream_use=lambda: None))
        actual = moe._MemoryBalancedEPMergedFc1GroupGemm.apply(*args, transfers)
    else:
        actual = moe._BalancedEPMergedFc1GroupGemm.apply(*args)
    actual.sum().backward()
    assert observed == [bound] * 4
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(tokens.grad, expected_dx, rtol=0, atol=0)


def _two_storage_worker(rank, rendezvous):
    import gc
    import weakref
    from types import SimpleNamespace

    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_dispatch
    from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
    from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe
    from veomni.distributed.moe.ep_buffer import ReplicaBufferPool

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    patch = pytest.MonkeyPatch()
    patch.setattr(moe, "group_gemm_same_nk", _reference_nk)
    patch.setattr(moe, "group_gemm_same_mn", _reference_mn)
    patch.setattr(ep_dispatch, "get_parallel_state", lambda: SimpleNamespace(ep_group=dist.group.WORLD))
    original_cat = torch.cat
    original_start = ep.ExpertReplicaTransfer.start
    original_record = ep.ExpertReplicaTransfer.record_stream_use
    owners = []
    recorded = []

    def tracking_start(*args, **kwargs):
        owner = original_start(*args, **kwargs)
        owners.append(weakref.ref(owner))
        return owner

    def no_weight_cat(tensors, *args, **kwargs):
        assert not any(t.ndim == 3 for t in tensors), "optimized production concatenated expert weights"
        return original_cat(tensors, *args, **kwargs)

    def tracking_record(owner):
        recorded.append(id(owner))
        original_record(owner)

    def run():
        return moe.dispatch_to_ep_class(
            moe.EPMergedFc1GroupGemm,
            2,
            route,
            selected,
            x,
            gate,
            down,
            limit,
            load_balancer=balancer,
        )

    patch.setattr(ep.ExpertReplicaTransfer, "start", tracking_start)
    patch.setattr(ep.ExpertReplicaTransfer, "record_stream_use", tracking_record)
    try:
        for balanced in (False, True):
            for limit in (None, 0.4):  # noqa: B007 - consumed by the run closure
                for train_gate, train_down in ((True, True), (True, False), (False, True), (False, False)):
                    torch.manual_seed(71 + rank)
                    selected = torch.full((3, 1), rank if balanced else 0, dtype=torch.long)
                    route = torch.tensor([[0.25], [0.5], [1.0]])
                    x = torch.randn(3, 3, requires_grad=True)
                    gate = torch.randn(1, 4, 3, requires_grad=train_gate)
                    down = torch.randn(1, 3, 2, requires_grad=train_down)
                    pool = ReplicaBufferPool()
                    balancer = SimpleNamespace(
                        ep_group=dist.group.WORLD,
                        layer_index=0,
                        optimize_memory=False,
                        buffer_pool=pool,
                        build_plan=lambda ids: ep.build_ep_balance_plan(ids, 2, dist.group.WORLD, 1),
                    )

                    reference = run()
                    reference.sum().backward()
                    expected = [t.grad.clone() if t.requires_grad else None for t in (x, gate, down)]
                    for t in (x, gate, down):
                        t.grad = None
                    owners.clear()
                    recorded.clear()
                    balancer.optimize_memory = True
                    with patch.context() as scoped:
                        scoped.setattr(torch, "cat", no_weight_cat)
                        first, second = run(), run()
                        torch.testing.assert_close(first, reference)
                        torch.testing.assert_close(second, reference)
                        if not balanced:
                            assert len(owners) == 4 and all(ref() is not None for ref in owners)
                            assert all(ref()._lease is not None for ref in owners)
                            assert len(recorded) == 4
                            pointers = [ref()._replica_weight.data_ptr() for ref in owners]
                            assert len(set(pointers)) == 4, "overlapping graphs reused live replica storage"
                        for output in (first, first, second):
                            output.sum().backward(retain_graph=True)
                            for t, want in zip((x, gate, down), expected, strict=True):
                                if want is None:
                                    assert t.grad is None
                                else:
                                    torch.testing.assert_close(t.grad, want)
                                t.grad = None
                        if not balanced:
                            assert all(ref() is not None for ref in owners), "backward released a live graph lease"
                            assert len(recorded) == 10
                        del output, first, second
                        gc.collect()
                        assert all(ref() is None for ref in owners)
                        if not balanced:
                            assert len(pool._idle) == 2
                        # A fresh no-grad forward must refresh cached storage after an optimizer-like update.
                        with torch.no_grad():
                            gate.add_(0.1)
                            fresh = run()
                        assert all(ref() is None for ref in owners)
                        if not balanced:
                            assert len(recorded) == 12
                    balancer.optimize_memory = False
                    with torch.no_grad():
                        torch.testing.assert_close(fresh, run())

        # Noncontiguous original parameters preserve the old valid behavior:
        # fallback does not borrow pool leases and may concatenate weights.
        selected = torch.zeros((3, 1), dtype=torch.long)
        gate = torch.randn(1, 3, 4).transpose(1, 2).requires_grad_()
        down = torch.randn(1, 2, 3).transpose(1, 2).requires_grad_()
        balancer.optimize_memory = False
        reference = run()
        balancer.optimize_memory = True
        owners.clear()
        result = run()
        torch.testing.assert_close(result, reference)
        assert len(owners) == 2 and all(ref() is None or ref()._lease is None for ref in owners)

        # A failed forward/backward still records any already-submitted reads.
        gate, down = gate.detach().contiguous().requires_grad_(), down.detach().contiguous().requires_grad_()
        recorded.clear()
        calls = 0

        def fail_second_gemm(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected GEMM submission failure")
            return _reference_nk(*args, **kwargs)

        with patch.context() as scoped:
            scoped.setattr(moe, "group_gemm_same_nk", fail_second_gemm)
            with pytest.raises(RuntimeError, match="injected GEMM"):
                run()
        assert len(recorded) == 2
        result = run()
        recorded.clear()

        def fail_gemm(*args, **kwargs):
            raise RuntimeError("injected GEMM submission failure")

        with patch.context() as scoped:
            scoped.setattr(moe, "group_gemm_same_nk", fail_gemm)
            with pytest.raises(RuntimeError, match="injected GEMM"):
                result.sum().backward()
        assert len(recorded) == 2

        # Compare the real BF16 production autograd to the existing fused
        # concatenated boundary (not CPU dispatch's separate BF16 Cat edge).
        plan = ep.build_ep_balance_plan(selected, 2, dist.group.WORLD, 1)
        cumsum = plan.tokens_per_local_physical_expert.sum(0).cumsum(0)
        for train_gate, train_down in ((True, True), (True, False), (False, True), (False, False)):
            x = torch.randn(3, 3, dtype=torch.bfloat16, requires_grad=True)
            gate = torch.randn(1, 4, 3, dtype=torch.bfloat16, requires_grad=train_gate)
            down = torch.randn(1, 3, 2, dtype=torch.bfloat16, requires_grad=train_down)
            transfers = tuple(original_start(w, plan, dist.group.WORLD, pool=pool) for w in (gate, down))
            replicas = tuple(owner.wait() for owner in transfers)
            args = (x, cumsum, gate, down, *replicas, plan, dist.group.WORLD, 0.4)
            baseline = moe._BalancedEPMergedFc1GroupGemm.apply(*args)
            baseline.sum().backward()
            expected = [t.grad.clone() if t.requires_grad else None for t in (x, gate, down)]
            for t in (x, gate, down):
                t.grad = None
            with patch.context() as scoped:
                scoped.setattr(torch, "cat", no_weight_cat)
                actual = moe._MemoryBalancedEPMergedFc1GroupGemm.apply(*args, transfers)
                actual.sum().backward()
            assert torch.equal(actual, baseline)
            for t, want in zip((x, gate, down), expected, strict=True):
                if want is None:
                    assert t.grad is None
                else:
                    assert torch.equal(t.grad, want)
    finally:
        patch.undo()
        dist.destroy_process_group()


def test_two_storage_production_autograd_parity_and_graph_lifetimes(tmp_path):
    mp.spawn(_two_storage_worker, args=(f"file://{tmp_path / 'two-storage'}",), nprocs=2, join=True)


def _recording_failure_worker(rank, rendezvous, consumer, stage):
    from types import SimpleNamespace

    if consumer == "gpu":
        from tests.parallel.test_moe_ep_load_balance_dispatch import ep_dispatch
        from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
        from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe
    else:
        from tests.ops.test_npu_moe_ep_load_balance import ep_dispatch
        from tests.ops.test_npu_moe_ep_load_balance import ep_load_balance as ep
        from tests.ops.test_npu_moe_ep_load_balance import npu_moe as moe

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    patch = pytest.MonkeyPatch()
    patch.setattr(ep_dispatch, "get_parallel_state", lambda: SimpleNamespace(ep_group=dist.group.WORLD))
    if consumer == "gpu":
        patch.setattr(moe, "group_gemm_same_nk", _reference_nk)
        patch.setattr(moe, "group_gemm_same_mn", _reference_mn)
    owners, attempts = [], []
    original_start = ep.ExpertReplicaTransfer.start
    original_record = ep.ExpertReplicaTransfer.record_stream_use
    fail_recording = stage == "forward"

    def start(*args, **kwargs):
        owner = original_start(*args, **kwargs)
        owners.append(owner)
        return owner

    def record(owner):
        index = owners.index(owner)
        attempts.append(index)
        if fail_recording and index == 0:
            raise RuntimeError("injected first-owner recording failure")
        original_record(owner)

    patch.setattr(ep.ExpertReplicaTransfer, "start", start)
    patch.setattr(ep.ExpertReplicaTransfer, "record_stream_use", record)
    try:
        selected = torch.zeros((2, 1), dtype=torch.long)
        route = torch.ones(2, 1)
        tokens = torch.randn(2, 3, requires_grad=True)
        gate = torch.randn(1, 4, 3, requires_grad=True)
        down = torch.randn(1, 3, 2, requires_grad=True)
        balancer = SimpleNamespace(
            ep_group=dist.group.WORLD,
            layer_index=0,
            optimize_memory=True,
            buffer_pool=ep.ReplicaBufferPool(),
            build_plan=lambda ids: ep.build_ep_balance_plan(ids, 2, dist.group.WORLD, 1),
        )

        def forward():
            if consumer == "gpu":
                return moe.dispatch_to_ep_class(
                    moe.EPMergedFc1GroupGemm,
                    2,
                    route,
                    selected,
                    tokens,
                    gate,
                    down,
                    None,
                    load_balancer=balancer,
                )
            return moe._npu_ep_balanced_moe_forward(
                2,
                route,
                selected,
                tokens,
                None,
                None,
                down,
                gate,
                dist.group.WORLD,
                None,
                balancer,
            )

        if stage == "forward":
            with pytest.raises(RuntimeError, match="first-owner recording failure"):
                forward()
        else:
            output = forward()
            attempts.clear()
            fail_recording = True
            with pytest.raises(RuntimeError, match="first-owner recording failure"):
                output.sum().backward()
        # This catches skipped recording, not device event correctness: both
        # real transfers have waited and both production consumers submitted work.
        assert all(owner._completed for owner in owners)
        # NPU has one autograd boundary per projection and records down before
        # gate during backward; GPU records the two owners together. Both must
        # still be attempted exactly once, even when gate recording raises.
        expected = [1, 0] if consumer == "npu" and stage == "backward" else [0, 1]
        assert attempts == expected, "first recording failure skipped a live storage owner"
    finally:
        patch.undo()
        dist.destroy_process_group()


@pytest.mark.parametrize("consumer", ["gpu", "npu"])
@pytest.mark.parametrize("stage", ["forward", "backward"])
def test_both_replica_owners_record_when_first_recording_raises(tmp_path, consumer, stage):
    mp.spawn(
        _recording_failure_worker,
        args=(f"file://{tmp_path / 'recording-failure'}", consumer, stage),
        nprocs=2,
        join=True,
    )
