"""Model-qualified opt-in checks for combined Ulysses and context parallelism."""

from types import SimpleNamespace

import pytest

import veomni.models.auto as auto
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import AcceleratorConfig


@pytest.mark.parametrize("cp_size,dp_shard", [(2, 2), (4, 1)])
def test_hybrid_topology_requires_an_explicit_opt_in(monkeypatch, cp_size, dp_shard):
    monkeypatch.setenv("WORLD_SIZE", "32")
    kwargs = dict(ulysses_size=8, cp_size=cp_size, dp_shard_size=dp_shard, dp_replicate_size=1)
    with pytest.raises(NotImplementedError, match="cannot be combined"):
        AcceleratorConfig(**kwargs)
    config = AcceleratorConfig(**kwargs, allow_hybrid_cp=True)
    assert config.dp_size == dp_shard
    assert config.ulysses_size * config.cp_size * config.dp_size == 32


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_hybrid_opt_in_rejects_non_booleans_before_topology_resolution(monkeypatch, value):
    def unexpected_topology_resolution(self):
        pytest.fail("Invalid hybrid opt-in must fail before topology resolution")

    monkeypatch.setattr(AcceleratorConfig, "_resolve_topology", unexpected_topology_resolution)
    with pytest.raises(ValueError, match="allow_hybrid_cp must be a boolean"):
        AcceleratorConfig(allow_hybrid_cp=value)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
@pytest.mark.parametrize("constructor", ["state", "init"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_direct_hybrid_state_rejects_non_booleans_before_mesh_creation(monkeypatch, value, constructor):
    import veomni.distributed.parallel_state as parallel_state

    def unexpected_mesh(*args, **kwargs):
        pytest.fail("Invalid hybrid opt-in must fail before mesh creation")

    monkeypatch.setattr(parallel_state, "init_device_mesh", unexpected_mesh)
    with pytest.raises(ValueError, match="allow_hybrid_cp must be a boolean"):
        if constructor == "state":
            parallel_state.ParallelState(allow_hybrid_cp=value)
        else:
            parallel_state._init_parallel_state(allow_hybrid_cp=value, name=None, device_type="cpu")


def test_false_hybrid_opt_in_preserves_single_axis_topology(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "32")
    config = AcceleratorConfig(ulysses_size=8, allow_hybrid_cp=False)
    assert config.dp_size == 4
    assert config.cp_size == 1


@pytest.mark.parametrize("npu", [False, True])
def test_hybrid_model_guard_keeps_other_families_closed(monkeypatch, npu):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, ulysses_size=8))
    monkeypatch.setattr(auto, "IS_NPU_AVAILABLE", npu)
    auto.check_context_parallel_supported(SimpleNamespace(model_type="qwen4_exp"))
    with pytest.raises(NotImplementedError, match="Qwen4-Exp only"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type="deepseek_v4"))


def test_native_cp_qualification_is_not_extended_to_other_models(monkeypatch):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, ulysses_size=1))
    monkeypatch.setattr(auto, "IS_NPU_AVAILABLE", True)
    with pytest.raises(NotImplementedError, match="Ascend/NPU"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type="deepseek_v4"))


def _hybrid_mesh_worker(rank, rendezvous):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=45)
    )
    try:
        state = _init_parallel_state(
            dp_size=1, cp_size=2, ulysses_size=2, allow_hybrid_cp=True, device_type="cpu", name="hybrid"
        )
        assert state.sp_rank == rank and state.sp_size == 4
        assert dist.get_process_group_ranks(state.sp_group) == [0, 1, 2, 3]
        assert dist.get_process_group_ranks(state.cp_group) == [rank % 2, rank % 2 + 2]
        assert dist.get_process_group_ranks(state.ulysses_group) == [rank // 2 * 2, rank // 2 * 2 + 1]
        local_context = torch.arange(rank * 3, (rank + 1) * 3)
        gathered = [torch.empty_like(local_context) for _ in range(4)]
        dist.all_gather(gathered, local_context, group=state.sp_group)
        torch.testing.assert_close(torch.cat(gathered), torch.arange(12))
        vision = _init_parallel_state(dp_size=1, ulysses_size=4, device_type="cpu", name="vision")
        assert vision.sp_rank == state.sp_rank
        assert dist.get_process_group_ranks(vision.sp_group) == dist.get_process_group_ranks(state.sp_group)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_hybrid_real_mesh_matches_contiguous_context_and_vision_layout(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_hybrid_mesh_worker, args=(str(tmp_path / "hybrid-mesh"),), nprocs=4, join=True)
