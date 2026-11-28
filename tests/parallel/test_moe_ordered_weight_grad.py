"""Ordered replica boundaries and seeded GEMM tail regressions."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.parallel.test_moe_ep_load_balance_planner import (
    _assert_global_invariants,
    _selected_rows,
)
from tests.parallel.test_moe_ep_load_balance_planner import (
    ep_load_balance as ep,
)
from veomni.distributed.moe import ep_grad
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device


_CPU_2_RANKS = LaunchSpec(nproc=2, device="cpu")
_CPU_4_RANKS = LaunchSpec(nproc=4, device="cpu")


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


def test_quantized_top8_plan_keeps_a_feasible_whole_tile_transfer():
    totals = [64] + [16] * 7 + [14] * 24
    rows = [[] for _ in range(64)]
    cursor = 0
    for expert, occurrences in enumerate(totals):
        for offset in range(occurrences):
            rows[(cursor + offset) % len(rows)].append(expert)
        cursor += occurrences
    assert all(len(row) == len(set(row)) == 8 for row in rows)
    selections = torch.tensor(rows).chunk(4)
    counts = torch.stack([torch.bincount(ids.flatten(), minlength=len(totals)) for ids in selections])
    plans = [
        ep._build_ep_balance_plan_from_counts(ids, counts, len(totals), rank, 1, token_granularity=32)
        for rank, ids in enumerate(selections)
    ]
    _assert_global_invariants(counts.tolist(), plans)
    assert plans[0].rank_loads_before == (176, 112, 112, 112)
    assert plans[0].rank_loads_after == (144, 144, 112, 112)
    assert [(r.source_expert, r.target_rank, r.moved_tokens) for r in plans[0].replicas] == [(0, 1, 32)]


def test_quantized_plan_recovers_a_tile_with_only_one_possible_recipient():
    counts = [[80, 0, 56, 0]] * 2
    plans = [
        ep._build_ep_balance_plan_from_counts(ids, torch.tensor(counts), 4, rank, 1, token_granularity=32)
        for rank, ids in enumerate(_selected_rows(counts))
    ]
    _assert_global_invariants(counts, plans)
    assert plans[0].rank_loads_before == (160, 112)
    assert plans[0].rank_loads_after == (128, 144)
    assert [replica.moved_tokens for replica in plans[0].replicas] == [32]


@pytest.mark.parametrize(
    ("counts", "loads"),
    [([[16, 8, 8, 8]] * 4, (64, 32, 32, 32)), ([[15, 15, 0, 0]] * 2, (60, 0))],
)
def test_quantized_plan_keeps_identity_without_an_improving_whole_tile(counts, loads):
    plan = ep._build_ep_balance_plan_from_counts(
        _selected_rows(counts)[0], torch.tensor(counts), 4, 0, 1, token_granularity=32
    )
    assert not plan.replicas
    assert plan.rank_loads_after == plan.rank_loads_before == loads


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


@pytest.mark.launch_plan(provider=_CPU_2_RANKS)
def test_kernel_metadata_collectively_rejects_different_tiles_and_signatures(tmp_path):
    mp.spawn(_metadata_worker, args=(f"file://{tmp_path / 'metadata'}",), nprocs=_CPU_2_RANKS.nproc, join=True)


def _prefix_worker(rank, rendezvous):
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

            run = ep_grad.make_ordered_weight_grad(plan, dist.group.WORLD, torch.device("cpu"), launcher=launcher)
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


@pytest.mark.launch_plan(provider=_CPU_4_RANKS)
def test_ordered_fp32_prefix_chain_matches_original_with_cancellation_and_tails(tmp_path):
    mp.spawn(_prefix_worker, args=(f"file://{tmp_path / 'prefix'}",), nprocs=_CPU_4_RANKS.nproc, join=True)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA metadata synchronization regression")
@pytest.mark.parametrize("side_stream", [False, True])
def test_cuda_ordered_metadata_does_not_synchronize_the_current_stream(monkeypatch, side_stream):
    from veomni.distributed.moe.ep_grad import make_ordered_weight_grad

    counts = [[128, 0, 0, 0], [128, 0, 0, 0]]
    plan = ep._build_ep_balance_plan_from_counts(
        _selected_rows(counts)[0], torch.tensor(counts), 4, 0, 1, token_granularity=32
    )
    assert plan.replicas
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    api = get_torch_device()
    device = torch.device(get_device_type())
    stream = api.Stream() if side_stream else api.current_stream()
    with api.stream(stream):
        # Initialize CUDA and profiler state outside the measured constructor.
        make_ordered_weight_grad(plan, None, device)
        api.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            with torch.profiler.record_function("ordered_ep_metadata"):
                make_ordered_weight_grad(plan, None, device)
            api.synchronize()
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


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA Triton kernel test")
@pytest.mark.parametrize("m,n", [(3, 5), (33, 65), (64, 64), (129, 63)])
@pytest.mark.parametrize("suffix", [0, 1, 31, 33])
def test_cuda_seeded_weight_gradient_preserves_original_order_and_masks_tails(m, n, suffix):
    from veomni.distributed.moe.ep_grad import _launch_weight_grad

    torch.manual_seed(41)
    device = get_device_type()
    count = 32 + suffix
    a = torch.randint(-2, 3, (count, m), device=device).to(torch.bfloat16)
    b = torch.randint(-2, 3, (count, n), device=device).to(torch.bfloat16)
    expected = a.float().T @ b.float()
    c = torch.empty((1, m, n), device=device, dtype=torch.float32)
    _launch_weight_grad(a[:32], b[:32], c, torch.tensor([32], device=device), count, False)
    _launch_weight_grad(a[32:], b[32:], c, torch.tensor([suffix], device=device), count, True)
    torch.testing.assert_close(c[0], expected, rtol=0, atol=0)
