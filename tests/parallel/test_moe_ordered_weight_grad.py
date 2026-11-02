"""Ordered replica boundaries and seeded GEMM tail regressions."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.parallel.test_moe_ep_load_balance_planner import (
    _assert_global_invariants,
    _selected_rows,
)
from tests.parallel.test_moe_ep_load_balance_planner import (
    ep_load_balance as ep,
)


@pytest.mark.parametrize("total", [0, 1, 31, 32, 33, 127, 128, 129, 251, 8193])
@pytest.mark.parametrize("granularity", [1, 32, 64])
def test_quantized_plan_conserves_occurrences_and_improves_load(total, granularity):
    counts = [[total // 4 + int(rank < total % 4), 0, 0, 0] for rank in range(4)]
    plans = [
        ep._build_ep_balance_plan_from_counts(
            selected, torch.tensor(counts), 4, rank, 1, token_granularity=granularity
        )
        for rank, selected in enumerate(_selected_rows(counts))
    ]
    _assert_global_invariants(counts, plans)
    assert all(replica.moved_tokens % granularity == 0 for replica in plans[0].replicas)
    if plans[0].replicas:
        assert sorted(plans[0].rank_loads_after, reverse=True) < sorted(plans[0].rank_loads_before, reverse=True)


@pytest.mark.parametrize("granularity", [0, -1, 1.5, True])
def test_invalid_weight_gradient_tile_granularity(granularity):
    with pytest.raises(ValueError, match="granularity"):
        ep._build_ep_balance_plan_from_counts(
            torch.zeros(2, 1, dtype=torch.long),
            torch.tensor([[2, 0], [2, 0]]),
            2,
            0,
            1,
            token_granularity=granularity,
        )


def _metadata_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        gathered = ep._gather_counts_with_kernel_metadata(torch.tensor([rank, 5]), dist.group.WORLD, (32, 7))
        assert torch.equal(gathered, torch.tensor([[0, 5], [1, 5]]))
        for metadata in ((32 * (rank + 1), 7), (32, 7 + rank)):
            with pytest.raises(ValueError, match="identical kernel configurations"):
                ep._gather_counts_with_kernel_metadata(torch.tensor([rank, 5]), dist.group.WORLD, metadata)
    finally:
        dist.destroy_process_group()


def test_kernel_metadata_collectively_rejects_different_tiles_and_signatures(tmp_path):
    mp.spawn(_metadata_worker, args=(f"file://{tmp_path / 'metadata'}",), nprocs=2, join=True)


def _prefix_worker(rank, rendezvous):
    # The CPU reference does not need to import accelerator kernel packages.
    spec = importlib.util.spec_from_file_location("ordered_ep_grad_test", Path(ep.__file__).with_name("ep_grad.py"))
    grad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grad)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        for total in (0, 63, 128, 131):
            counts = torch.zeros(4, 4, dtype=torch.long)
            counts[:, 0] = torch.tensor([total // 4 + int(r < total % 4) for r in range(4)])
            plan = ep._build_ep_balance_plan_from_counts(
                torch.zeros(int(counts[rank, 0]), 1, dtype=torch.long),
                counts,
                4,
                rank,
                1,
                token_granularity=32,
            )
            torch.manual_seed(73)
            full_a = torch.randn(total, 3).bfloat16()
            full_b = torch.randn(total, 5).bfloat16()
            if total == 128:
                full_a.zero_()
                full_b.fill_(1)
                full_a[0].fill_(2**60)
                full_a[32].fill_(1)
                full_a[96].fill_(-(2**60))
            sizes = plan.tokens_per_local_physical_expert.sum(0)
            pieces = [full_a[:0] for _ in sizes]
            other = [full_b[:0] for _ in sizes]
            cursor = 0
            for replica in plan.replicas:
                end = cursor + replica.moved_tokens
                if replica.target_rank == rank:
                    pieces[1 + replica.target_slot] = full_a[cursor:end]
                    other[1 + replica.target_slot] = full_b[cursor:end]
                cursor = end
            if rank == 0:
                pieces[0], other[0] = full_a[cursor:], full_b[cursor:]
            a, b = torch.cat(pieces), torch.cat(other)
            assert [len(v) for v in pieces] == sizes.tolist()
            c = torch.full((len(sizes), 3, 5), float("nan"))

            def reference(aa, bb, seed=None):
                acc = torch.zeros(3, 5) if seed is None else seed.clone()
                for start in range(0, len(aa), 32):
                    acc.add_(aa[start : start + 32].float().T @ bb[start : start + 32].float())
                return acc

            def launcher(aa, bb, cc, ends, original_total, accumulate, *, group_indices=None):
                start = 0
                active = set(range(len(ends))) if group_indices is None else set(group_indices.tolist())
                for group, end in enumerate(ends.tolist()):
                    if group in active:
                        cc[group].copy_(reference(aa[start:end], bb[start:end], cc[group] if accumulate else None))
                    start = end

            run = grad.make_ordered_weight_grad(plan, dist.group.WORLD, torch.device("cpu"), launcher=launcher)
            run(a, b, c, sizes.cumsum(0), len(a), transpose_a=True)
            expected = reference(full_a, full_b) if rank == 0 else torch.zeros(3, 5)
            torch.testing.assert_close(c[0], expected, rtol=0, atol=0)
            if total == 128:
                wrong_order = torch.tensor(-(2**60), dtype=torch.float32)
                for value in (2**60, 1, 0):
                    wrong_order.add_(value)
                assert wrong_order.item() == 1 and not torch.count_nonzero(expected)
    finally:
        dist.destroy_process_group()


def test_ordered_fp32_prefix_chain_matches_original_with_cancellation_and_tails(tmp_path):
    mp.spawn(_prefix_worker, args=(f"file://{tmp_path / 'prefix'}",), nprocs=4, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata synchronization regression")
@pytest.mark.parametrize("side_stream", [False, True])
def test_cuda_ordered_metadata_does_not_synchronize_the_current_stream(monkeypatch, side_stream):
    from veomni.distributed.moe.ep_grad import make_ordered_weight_grad

    counts = [[128, 0, 0, 0], [128, 0, 0, 0]]
    plan = ep._build_ep_balance_plan_from_counts(
        _selected_rows(counts)[0], torch.tensor(counts), 4, 0, 1, token_granularity=32
    )
    assert plan.replicas
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        # Initialize CUDA and profiler state outside the measured constructor.
        make_ordered_weight_grad(plan, None, torch.device("cuda"))
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            with torch.profiler.record_function("ordered_ep_metadata"):
                make_ordered_weight_grad(plan, None, torch.device("cuda"))
            torch.cuda.synchronize()
    events = profile.events()
    scope = next(event for event in events if event.name == "ordered_ep_metadata")
    synchronizations = [
        event
        for event in events
        if event.name == "cudaStreamSynchronize"
        and scope.time_range.start <= event.time_range.start < scope.time_range.end
        and event.thread == scope.thread
    ]
    assert not synchronizations, "Ordered metadata upload must not fence preceding GPU work on the host"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Triton kernel test")
@pytest.mark.parametrize("m,n", [(3, 5), (33, 65), (64, 64), (129, 63)])
@pytest.mark.parametrize("suffix", [0, 1, 31, 33])
def test_cuda_seeded_weight_gradient_preserves_original_order_and_masks_tails(m, n, suffix):
    from veomni.distributed.moe.ep_grad import _launch_weight_grad

    torch.manual_seed(41)
    count = 32 + suffix
    a = torch.randint(-2, 3, (count, m), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (count, n), device="cuda").to(torch.bfloat16)
    expected = a.float().T @ b.float()
    c = torch.empty((1, m, n), device="cuda", dtype=torch.float32)
    _launch_weight_grad(a[:32], b[:32], c, torch.tensor([32], device="cuda"), count, False)
    _launch_weight_grad(a[32:], b[32:], c, torch.tensor([suffix], device="cuda"), count, True)
    torch.testing.assert_close(c[0], expected, rtol=0, atol=0)
