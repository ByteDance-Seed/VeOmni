# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Exact global and local tensor layout for composed EP/FSDP checkpoint shards."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import (
    _EXTRA_PARALLEL_LAYOUT_MARKER,
    DistributedCheckpointer,
    ModelState,
    OptimizerState,
    _apply_extra_parallel_dim,
    _export_extra_parallel_layout_key,
    _extra_parallel_layout_key,
    _get_sharding_plan,
    _ModelStrictLoadPlanner,
    _needs_extra_parallel_layout_marker,
    drop_extra_parallel_dim_v2,
    restore_extra_parallel_dim,
    restore_extra_parallel_dim_v2,
)
from veomni.distributed.parallel_plan import SpecInfo


def _roundtrip(rank, root, ep_shard_dim, fsdp_shard_dim):
    dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
        fsdp_mesh = mesh["fsdp"]
        full = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        ep_index = mesh.get_coordinate()[1]
        local_ep = torch.chunk(full, 2, dim=ep_shard_dim)[ep_index]
        local = distribute_tensor(local_ep.contiguous(), fsdp_mesh, placements=[Shard(fsdp_shard_dim)])
        with pytest.raises(RuntimeError, match="declared FSDP placement"):
            restore_extra_parallel_dim_v2(
                local, mesh, fsdp_mesh, ep_shard_dim, (fsdp_shard_dim + 1) % 2, global_shape=tuple(full.shape)
            )
        combined = restore_extra_parallel_dim_v2(
            local, mesh, fsdp_mesh, ep_shard_dim, fsdp_shard_dim, global_shape=tuple(full.shape)
        )
        torch.testing.assert_close(combined.full_tensor(), full, rtol=0, atol=0)
        spec = SpecInfo("ep", Shard(ep_shard_dim), "weight", mesh, fsdp_shard_dim, None, tuple(full.shape), True)
        parallel = SimpleNamespace(extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh})
        converted = _apply_extra_parallel_dim(
            {"weight": local}, {"weight": spec}, parallel, "restore", key_match="exact"
        )
        torch.testing.assert_close(converted["weight"].full_tensor(), full, rtol=0, atol=0)
        spec.checkpoint_layout_v2 = False
        legacy = restore_extra_parallel_dim(local, mesh, fsdp_mesh, ep_shard_dim, fsdp_shard_dim)
        actual = _apply_extra_parallel_dim({"weight": local}, {"weight": spec}, parallel, "restore", key_match="exact")
        assert actual["weight"].shape == legacy.shape
        assert actual["weight"].placements == legacy.placements
        torch.testing.assert_close(actual["weight"].to_local(), legacy.to_local(), rtol=0, atol=0)
        dcp.save({"expert": combined}, checkpoint_id=f"{root}/checkpoint")
        target = restore_extra_parallel_dim_v2(
            torch.zeros_like(local), mesh, fsdp_mesh, ep_shard_dim, fsdp_shard_dim, global_shape=tuple(full.shape)
        )
        dcp.load({"expert": target}, checkpoint_id=f"{root}/checkpoint")
        torch.testing.assert_close(target.full_tensor(), full, rtol=0, atol=0)
        restored = drop_extra_parallel_dim_v2(target, fsdp_mesh, fsdp_shard_dim, extra_parallel_name="ep")
        assert restored.shape == local.shape
        assert restored.placements == local.placements
        torch.testing.assert_close(restored.to_local(), local.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
@pytest.mark.parametrize("ep_shard_dim,fsdp_shard_dim", [(0, 0), (0, 1), (-1, -1), (-1, 1)])
def test_composed_ep_fsdp_layout_roundtrip(tmp_path, ep_shard_dim, fsdp_shard_dim):
    torch.multiprocessing.spawn(_roundtrip, args=(str(tmp_path), ep_shard_dim, fsdp_shard_dim), nprocs=4, join=True)


def _hsdp_roundtrip(rank, root):
    dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=rank, world_size=8)
    try:
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("replicate", "fsdp", "ep"))
        fsdp_mesh = mesh["replicate", "fsdp"]
        full = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        ep_index = mesh.get_coordinate()[-1]
        local_ep = torch.chunk(full, 2, dim=0)[ep_index]
        local = distribute_tensor(local_ep.contiguous(), fsdp_mesh, placements=[Replicate(), Shard(0)])
        combined = restore_extra_parallel_dim_v2(local, mesh, fsdp_mesh, 0, 0, global_shape=tuple(full.shape))
        torch.testing.assert_close(combined.full_tensor(), full, rtol=0, atol=0)
        dcp.save({"expert": combined}, checkpoint_id=f"{root}/checkpoint")
        target = restore_extra_parallel_dim_v2(
            torch.zeros_like(local), mesh, fsdp_mesh, 0, 0, global_shape=tuple(full.shape)
        )
        dcp.load({"expert": target}, checkpoint_id=f"{root}/checkpoint")
        resumed = drop_extra_parallel_dim_v2(target, fsdp_mesh, 0, extra_parallel_name="ep")
        torch.testing.assert_close(resumed.to_local(), local.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=8))
def test_composed_ep_hsdp_layout_roundtrip(tmp_path):
    torch.multiprocessing.spawn(_hsdp_roundtrip, args=(str(tmp_path),), nprocs=8, join=True)


def _cross_layout_roundtrip(rank, root, source_dim, target_dim):
    dist.init_process_group("gloo", init_method=f"file://{root}/init", rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
        fsdp_mesh = mesh["fsdp"]
        full = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        local_ep = torch.chunk(full, 2, dim=0)[mesh.get_coordinate()[1]]
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )

        def build(fsdp_dim, fill_zero):
            model = torch.nn.Module()
            value = torch.zeros_like(local_ep) if fill_zero else local_ep
            shard = distribute_tensor(value.contiguous(), fsdp_mesh, placements=[Shard(fsdp_dim)])
            model.register_parameter("weight", torch.nn.Parameter(shard))
            model._fqn2spec_info = {
                "weight": SpecInfo("ep", Shard(0), "weight", mesh, fsdp_dim, None, tuple(full.shape), True)
            }
            return model

        source = build(source_dim, False)
        dcp.save({"model": ModelState(source, parallel_state=parallel)}, checkpoint_id=f"{root}/checkpoint")
        target = build(target_dim, True)
        dcp.load(
            {"model": ModelState(target, parallel_state=parallel)},
            checkpoint_id=f"{root}/checkpoint",
            planner=_ModelStrictLoadPlanner(strict_model=True),
        )
        expected = distribute_tensor(local_ep.contiguous(), fsdp_mesh, placements=[Shard(target_dim)])
        torch.testing.assert_close(target.weight.to_local(), expected.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
@pytest.mark.parametrize("source_dim,target_dim", [(0, 1), (1, 0)])
def test_canonical_checkpoint_can_change_fsdp_shard_axis(tmp_path, source_dim, target_dim):
    torch.multiprocessing.spawn(
        _cross_layout_roundtrip, args=(str(tmp_path), source_dim, target_dim), nprocs=4, join=True
    )


def test_spec_info_positional_shard_dimension_remains_compatible():
    spec = SpecInfo("ep", Shard(0), "weight", None, 0)
    assert spec.fsdp_shard_dim == 0
    assert spec.global_shape is None
    assert not _needs_extra_parallel_layout_marker({"weight": spec})
    spec.checkpoint_layout_v2 = True
    spec.global_shape = (4, 4)
    assert _needs_extra_parallel_layout_marker({"weight": spec})


def test_negative_dimensions_are_canonicalized_for_layout_identity():
    mesh = SimpleNamespace(shape=(2, 2))
    negative = SpecInfo("ep", Shard(-1), "weight", mesh, 1, None, (4, 8), True)
    positive = SpecInfo("ep", Shard(1), "weight", mesh, -1, None, (4, 8), True)
    assert _needs_extra_parallel_layout_marker({"weight": negative})
    assert _extra_parallel_layout_key({"weight": negative}) == _extra_parallel_layout_key({"weight": positive})


def test_v2_requires_original_shape_and_fsdp_tensor():
    spec = SpecInfo("ep", Shard(0), "weight", None, 0, checkpoint_layout_v2=True)
    with pytest.raises(RuntimeError, match="requires the original global_shape"):
        _extra_parallel_layout_key({"weight": spec})


def test_v2_optimizer_fails_before_model_or_checkpoint_mutation(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    mesh = SimpleNamespace(shape=(2, 2))
    model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 0, None, (2, 2), True)}
    optimizer = SimpleNamespace()
    parallel = SimpleNamespace(dp_mode="fsdp2")
    adapter = OptimizerState(model, optimizer, parallel_state=parallel)
    with pytest.raises(RuntimeError, match="owner-aware checkpoint codec"):
        adapter.state_dict()
    with pytest.raises(RuntimeError, match="owner-aware checkpoint codec"):
        adapter.load_state_dict({})
    with pytest.raises(RuntimeError, match="owner-aware checkpoint codec"):
        DistributedCheckpointer.save(str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert not list(tmp_path.iterdir())


def _rank_local_invalid_layout_worker(rank, root, failure):
    from datetime import timedelta

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        model = torch.nn.Linear(2, 2, bias=False)
        mesh = SimpleNamespace(shape=(2, 2))
        shape = None if failure == "shape" and rank == 0 else (2, 2)
        v2 = not (failure == "version" and rank == 0)
        model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 0, None, shape, v2)}
        with pytest.raises(RuntimeError):
            DistributedCheckpointer.save(
                f"{root}/checkpoint", {"model": model}, parallel_state=SimpleNamespace(dp_mode="fsdp2")
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("failure", ["shape", "version"])
def test_rank_local_invalid_layout_stops_all_ranks_before_write(tmp_path, failure):
    torch.multiprocessing.spawn(_rank_local_invalid_layout_worker, args=(str(tmp_path), failure), nprocs=2, join=True)
    assert not (tmp_path / "checkpoint").exists()


def test_same_v2_wrapper_can_save_and_load(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    mesh = SimpleNamespace(shape=(2, 2))
    model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 0, None, (2, 2), True)}
    parallel = SimpleNamespace(dp_mode="fsdp2")
    monkeypatch.setattr(
        ModelState, "get_state_dict_with_extra_parallel_dim_preprocess", lambda self, state, action: state
    )
    wrapper = ModelState(model, parallel_state=parallel)
    saved = model.weight.detach().clone()
    direct = wrapper.state_dict()
    wrapper.load_state_dict(direct)
    torch.testing.assert_close(model.weight, saved, rtol=0, atol=0)
    dcp.save({"model": wrapper}, checkpoint_id=tmp_path)
    with torch.no_grad():
        model.weight.zero_()
    dcp.load({"model": wrapper}, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(strict_model=True))
    torch.testing.assert_close(model.weight, saved, rtol=0, atol=0)


def test_v2_tensors_cannot_be_silently_read_with_legacy_dcp_keys(tmp_path, monkeypatch):
    from torch.distributed.checkpoint.api import CheckpointException

    model = torch.nn.Linear(2, 2, bias=False)
    mesh = SimpleNamespace(shape=(2, 2))
    spec = SpecInfo("ep", Shard(0), "weight", mesh, 0, None, (2, 2), True)
    model._fqn2spec_info = {"weight": spec}
    monkeypatch.setattr(
        ModelState, "get_state_dict_with_extra_parallel_dim_preprocess", lambda self, state, action: state
    )
    wrapper = ModelState(model, parallel_state=SimpleNamespace(dp_mode="fsdp2"))
    dcp.save({"model": wrapper}, checkpoint_id=tmp_path)
    legacy = torch.zeros_like(model.weight)
    with pytest.raises(CheckpointException):
        dcp.load({"model": {"weight": legacy}}, checkpoint_id=tmp_path)
    torch.testing.assert_close(legacy, torch.zeros_like(legacy), rtol=0, atol=0)
    dcp.load(
        {"model": {"weight": legacy}}, checkpoint_id=tmp_path, planner=dcp.DefaultLoadPlanner(allow_partial_load=True)
    )
    torch.testing.assert_close(legacy, torch.zeros_like(legacy), rtol=0, atol=0)
    marker = _extra_parallel_layout_key({"weight": spec})
    assert _export_extra_parallel_layout_key(marker) is None
    assert _export_extra_parallel_layout_key(marker + "__model.foo.weight") == "model.foo.weight"


def test_legacy_partial_dcp_cannot_overwrite_v2_target(tmp_path, monkeypatch):
    dcp.save({"model": {"weight": torch.ones(2, 2)}}, checkpoint_id=tmp_path)
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    mesh = SimpleNamespace(shape=(2, 2))
    model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 0, None, (2, 2), True)}
    monkeypatch.setattr(
        ModelState, "get_state_dict_with_extra_parallel_dim_preprocess", lambda self, state, action: state
    )
    wrapper = ModelState(model, parallel_state=SimpleNamespace(dp_mode="fsdp2"))
    dcp.load({"model": wrapper}, checkpoint_id=tmp_path, planner=dcp.DefaultLoadPlanner(allow_partial_load=True))
    torch.testing.assert_close(model.weight, torch.zeros_like(model.weight), rtol=0, atol=0)


def test_v2_online_hf_export_restores_fqn_and_preserves_values(monkeypatch):
    from veomni.utils.save_safetensor_utils import get_model_save_state

    model = torch.nn.Linear(2, 2, bias=False)
    mesh = SimpleNamespace(shape=(2, 2))
    model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 0, None, (2, 2), True)}
    monkeypatch.setattr(
        ModelState, "get_state_dict_with_extra_parallel_dim_preprocess", lambda self, state, action: state
    )
    before = model.weight.detach().clone()
    exported = get_model_save_state(model, None, parallel_state=SimpleNamespace(dp_mode="fsdp2"))
    assert set(exported) == {"weight"}
    torch.testing.assert_close(exported["weight"], before.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def test_opted_in_layout_rejects_unversioned_checkpoint_before_tensor_copy(tmp_path):
    from torch.distributed.checkpoint.api import CheckpointException

    dcp.save({"model": {"weight": torch.ones(2)}}, checkpoint_id=tmp_path)
    target = torch.zeros(2)
    with pytest.raises(CheckpointException, match="Incompatible ExtraParallel checkpoint layout"):
        dcp.load(
            {"model": {"weight": target, _EXTRA_PARALLEL_LAYOUT_MARKER + "a" * 64: True}},
            checkpoint_id=tmp_path,
            planner=_ModelStrictLoadPlanner(strict_model=False),
        )
    torch.testing.assert_close(target, torch.zeros_like(target), rtol=0, atol=0)


def test_canonical_layout_rejects_mixed_legacy_shards():
    mesh = SimpleNamespace(shape=(2, 2))
    source_a = SpecInfo("ep", Shard(0), "a", mesh, 0, None, (4, 4), True)
    source_b = SpecInfo("ep", Shard(0), "b", mesh, 0, None, (4, 4), False)
    with pytest.raises(RuntimeError, match="cannot mix versioned and legacy"):
        _extra_parallel_layout_key({"a": source_a, "b": source_b})


def test_hf_export_rejects_duplicate_legacy_and_canonical_source(tmp_path):
    mesh = SimpleNamespace(shape=(2, 2))
    spec = SpecInfo("ep", Shard(0), "weight", mesh, 0, None, (2, 2), True)
    marker = _extra_parallel_layout_key({"weight": spec})
    dcp.save(
        {
            "model": {
                "model.weight": torch.ones(2, 2),
                marker + "__model.weight": torch.zeros(2, 2),
                marker: "ep-outer-v1",
            }
        },
        checkpoint_id=tmp_path,
    )
    with pytest.raises(RuntimeError, match="mixes versioned and legacy"):
        _get_sharding_plan(str(tmp_path))


def test_hf_export_rejects_versioned_tensor_without_marker(tmp_path):
    marker = _extra_parallel_layout_key({})
    dcp.save({"model": {marker + "__model.weight": torch.ones(2, 2)}}, checkpoint_id=tmp_path)
    with pytest.raises(RuntimeError, match="marker is missing"):
        _get_sharding_plan(str(tmp_path))


def test_hf_export_rejects_malformed_marker_value(tmp_path):
    marker = _extra_parallel_layout_key({})
    dcp.save({"model": {marker: "wrong", marker + "__model.weight": torch.ones(2)}}, checkpoint_id=tmp_path)
    with pytest.raises(RuntimeError, match="marker value"):
        _get_sharding_plan(str(tmp_path))


def test_hf_export_rejects_unknown_extra_parallel_layout():
    unknown = _EXTRA_PARALLEL_LAYOUT_MARKER + "0" * 64 + "__model.weight"
    with pytest.raises(RuntimeError, match="verified migration"):
        _export_extra_parallel_layout_key(unknown)


def test_v2_model_load_rejects_dtype_conversion(tmp_path):
    marker = _extra_parallel_layout_key({})
    key = marker + "__weight"
    dcp.save({"model": {marker: "ep-outer-v1", key: torch.ones(2, dtype=torch.float32)}}, checkpoint_id=tmp_path)
    state = {"model": {marker: "ep-outer-v1", key: torch.zeros(2, dtype=torch.bfloat16)}}
    with pytest.raises(BaseException, match="checkpoint dtype differs"):
        dcp.load(state, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(strict_model=True))
    assert torch.equal(state["model"][key], torch.zeros(2, dtype=torch.bfloat16))


def test_v2_model_load_rejects_legacy_keys_alongside_marker(tmp_path):
    marker = _extra_parallel_layout_key({})
    key = marker + "__weight"
    dcp.save(
        {"model": {marker: "ep-outer-v1", key: torch.ones(2), "weight": torch.full((2,), 2.0)}}, checkpoint_id=tmp_path
    )
    state = {"model": {marker: "ep-outer-v1", key: torch.zeros(2)}}
    with pytest.raises(BaseException, match="mixes versioned and legacy"):
        dcp.load(state, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(strict_model=True))
    assert torch.equal(state["model"][key], torch.zeros(2))


def test_v2_rejects_checkpoint_wrapper_fqn_not_in_canonical_state():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    model = torch.nn.Module()
    model.block = checkpoint_wrapper(torch.nn.Linear(2, 2, bias=False))
    mesh = SimpleNamespace(shape=(2, 2))
    name = "block._checkpoint_wrapped_module.weight"
    model._fqn2spec_info = {name: SpecInfo("ep", Shard(0), name, mesh, 1, None, (2, 2), True)}
    with pytest.raises(RuntimeError, match="not present in the canonical state dict"):
        ModelState(model, parallel_state=SimpleNamespace(dp_mode="fsdp2")).state_dict()


def test_v2_trainable_only_allows_frozen_spec_absent():
    from veomni.checkpoint.dcp_checkpointer import _check_canonical_spec_coverage

    model = torch.nn.Linear(2, 2)
    model.weight.requires_grad_(False)
    mesh = SimpleNamespace(shape=(2, 2))
    model._fqn2spec_info = {
        "weight": SpecInfo("ep", Shard(0), "weight", mesh, 1, None, (2, 2), True),
        "bias": SpecInfo("ep", Replicate(), "bias", mesh, 1),
    }
    _check_canonical_spec_coverage(model, model._fqn2spec_info, {"bias"}, trainable_only=True)
    with pytest.raises(RuntimeError, match="not present in the canonical state dict"):
        _check_canonical_spec_coverage(model, model._fqn2spec_info, {"bias"}, trainable_only=False)


def test_v2_rejects_missing_runtime_parameter_spec():
    model = torch.nn.Linear(2, 2)
    mesh = SimpleNamespace(shape=(2, 2))
    model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, 1, None, (2, 2), True)}
    with pytest.raises(RuntimeError, match="plan is missing parameter: bias"):
        ModelState(model, parallel_state=SimpleNamespace(dp_mode="fsdp2")).state_dict()


def _high_level_roundtrip_worker(rank, root):
    from datetime import timedelta

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init-high", rank=rank, world_size=4, timeout=timedelta(seconds=45)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp_mesh = mesh["ep_fsdp"]
        ep_rank = mesh.get_local_rank("ep")
        full = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        ep_local = full.chunk(2, dim=0)[ep_rank].contiguous()

        def model_for(axis, values):
            model = torch.nn.Module()
            model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(values, fsdp_mesh, [Shard(axis)])))
            model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, axis, None, (8, 8), True)}
            return model

        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )
        source = model_for(1, ep_local)
        DistributedCheckpointer.save(f"{root}/high", {"model": source}, parallel_state=parallel)
        target = model_for(0, torch.zeros_like(ep_local))
        DistributedCheckpointer.load(f"{root}/high", {"model": target}, parallel_state=parallel)
        expected = distribute_tensor(ep_local, fsdp_mesh, [Shard(0)])
        torch.testing.assert_close(target.weight.to_local(), expected.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_high_level_v2_cross_layout_model_only_roundtrip(tmp_path):
    torch.multiprocessing.spawn(_high_level_roundtrip_worker, args=(str(tmp_path),), nprocs=4, join=True)


def test_regular_nested_tensor_list_loads_with_strict_planner(tmp_path):
    dcp.save({"optimizer": {"foo": [torch.ones(2), torch.full((2,), 2.0)]}}, checkpoint_id=tmp_path)
    target = [torch.zeros(2), torch.zeros(2)]
    dcp.load({"optimizer": {"foo": target}}, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(False))
    torch.testing.assert_close(target[0], torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(target[1], torch.full((2,), 2.0), rtol=0, atol=0)
