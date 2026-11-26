"""The CUDA planner must match the count-driven CPU reference exactly."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_dist_comm_backend, get_torch_device


def _worker(rank, rendezvous):
    from veomni.distributed.moe.ep_grad import weight_grad_config_metadata
    from veomni.distributed.moe.ep_load_balance import (
        _build_ep_balance_plan_from_counts,
        build_ep_balance_plan,
    )

    device = torch.device(get_device_type(), rank)
    get_torch_device().set_device(device)
    dist.init_process_group(get_dist_comm_backend(), init_method=rendezvous, rank=rank, world_size=2)
    try:
        for rows in (0, 31, 128, 257):
            for mode in ("hot", "native"):
                torch.manual_seed(715 + rank)
                selected = torch.rand(rows, 16, device=device).topk(3, dim=-1).indices
                if mode == "hot":
                    selected = torch.arange(3, device=device).expand(rows, 3).contiguous()
                local = torch.bincount(selected.reshape(-1), minlength=16)
                counts = torch.empty(32, dtype=torch.long, device=device)
                dist.all_gather_into_tensor(counts, local)
                expected = _build_ep_balance_plan_from_counts(
                    selected.cpu(),
                    counts.reshape(2, 16).cpu(),
                    16,
                    rank,
                    2,
                    token_granularity=weight_grad_config_metadata()[0],
                )
                actual = build_ep_balance_plan(selected, 16, dist.group.WORLD, 2)
                assert actual.selected_physical_experts.device == selected.device
                assert actual.selected_physical_experts.dtype == torch.long
                torch.testing.assert_close(actual.selected_physical_experts.cpu(), expected.selected_physical_experts)
                torch.testing.assert_close(
                    actual.tokens_per_local_physical_expert, expected.tokens_per_local_physical_expert
                )
                for field in ("replicas", "input_splits", "output_splits", "rank_loads_before", "rank_loads_after"):
                    assert getattr(actual, field) == getattr(expected, field), field
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE or get_torch_device().device_count() < 2, reason="Requires two CUDA devices")
def test_cuda_host_planning_matches_reference(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'cuda_host_plan'}",), nprocs=2, join=True)
