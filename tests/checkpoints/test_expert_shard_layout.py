# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU/Gloo regressions for canonical EP-first DCP expert coordinates."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from scripts.ci.accelerator.launch_plan import LaunchSpec, from_parameter
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import (
    _EXPERT_LAYOUTS_KEY,
    DistributedCheckpointer,
    ModelState,
    _expert_layout,
    _extra_parallel_first_mesh,
    _extra_parallel_layout_key,
    _ExtraParallelSavePlanner,
    _get_sharding_plan,
    _ModelStrictLoadPlanner,
    _process_shard,
    dcp_to_torch_state_dict,
    drop_extra_parallel_dim,
    restore_extra_parallel_dim,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _layout_worker(rank, world, directory, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=180)
    )
    try:
        shape, names = ((2, 2), ("ep_fsdp", "ep")) if world == 4 else ((2, 2, 2), ("ep_replicate", "ep_fsdp", "ep"))
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
        submesh = mesh[names[:-1]]
        coordinate = mesh.get_coordinate()
        fsdp_rank, ep_rank = coordinate[-2:]
        # Unique global values expose expert swaps even if all local shapes agree.
        global_weight = torch.arange(8 * 16 * 4, dtype=torch.float32).reshape(8, 16, 4)
        ep_weight = global_weight.chunk(2, 0)[ep_rank]

        # EP-only rank-one values must ignore the unused default FSDP axis.
        vector = torch.arange(4, dtype=torch.float32) + ep_rank
        plain = restore_extra_parallel_dim(vector, mesh, submesh)
        torch.testing.assert_close(plain.to_local(), vector, rtol=0, atol=0)
        expected_vector = torch.cat((torch.arange(4), torch.arange(4) + 1)).float()
        torch.testing.assert_close(plain.full_tensor(), expected_vector, rtol=0, atol=0)
        vector_path = directory + "ep_only_vector"
        dcp.save({"model": {"experts.vector": plain}}, checkpoint_id=vector_path, planner=_ExtraParallelSavePlanner())
        target = restore_extra_parallel_dim(torch.zeros_like(vector), mesh, submesh)
        dcp.load(
            {"model": {"experts.vector": target}},
            checkpoint_id=vector_path,
            planner=_ModelStrictLoadPlanner(strict_model=True),
        )
        torch.testing.assert_close(target.to_local(), vector, rtol=0, atol=0)
        fsdp_vector = DTensor.from_local(vector, submesh, [Shard(0)] if world == 4 else [Replicate(), Shard(0)])
        with pytest.raises(RuntimeError, match="ExtraParallel shard dimension is outside"):
            restore_extra_parallel_dim(fsdp_vector, mesh, submesh)

        def wrap(dim, *, empty=False):
            local = ep_weight.chunk(2, dim)[fsdp_rank].contiguous()
            if empty:
                local = torch.zeros_like(local)
            placement = [Shard(dim)] if world == 4 else [Replicate(), Shard(dim)]
            local_dt = DTensor.from_local(local, submesh, placement)
            return restore_extra_parallel_dim(local_dt, mesh, submesh, 0, dim)

        for source_dim in (0, 1):
            source = wrap(source_dim)
            torch.testing.assert_close(source.full_tensor(), global_weight, rtol=0, atol=0)
            path = directory + str(source_dim)
            state = {"model": {"weight": source}, "optimizer": {"exp_avg": source.clone()}}
            dcp.save(state, checkpoint_id=path, planner=_ExtraParallelSavePlanner())
            metadata = dcp.FileSystemReader(path).read_metadata()
            layouts = metadata.planner_data[_EXPERT_LAYOUTS_KEY]
            assert layouts == {
                "version": 1,
                "tensors": {
                    key: {
                        "ep_shard_dim": 0,
                        "fsdp_shard_dim": source_dim,
                        "ep_size": 2,
                        "fsdp_size": 2,
                        "expert_order": "canonical",
                    }
                    for key in ("model.weight", "optimizer.exp_avg")
                },
            }
            target = {"model": {"weight": wrap(1 - source_dim, empty=True)}}
            dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
            loaded = target["model"]["weight"]
            torch.testing.assert_close(loaded.full_tensor(), global_weight, rtol=0, atol=0)
            local_dt = drop_extra_parallel_dim(loaded, submesh, 1 - source_dim)
            torch.testing.assert_close(local_dt.full_tensor(), ep_weight, rtol=0, atol=0)

        if world == 4:
            from veomni.distributed.parallel_plan import SpecInfo

            def model_for(values):
                model = torch.nn.Module()
                # FSDP2 allocates the shard storage independently.  A plain
                # contiguous slice may retain a nonzero offset into ``values``.
                local = values.chunk(2, 0)[fsdp_rank].clone()
                model.register_parameter("weight", torch.nn.Parameter(DTensor.from_local(local, submesh, [Shard(0)])))
                model._fqn2spec_info = {"weight": SpecInfo("ep", Shard(0), "weight", mesh, fsdp_shard_dim=0)}
                return model

            parallel = SimpleNamespace(
                dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
            )
            source = model_for(ep_weight)
            for save_async in (False, True):
                path = directory + f"prepared-high-level-{save_async}"
                DistributedCheckpointer.save(path, {"model": source}, save_async=save_async, parallel_state=parallel)
                DistributedCheckpointer.wait_for_pending_save()
                target = model_for(torch.zeros_like(ep_weight))
                DistributedCheckpointer.load(path, {"model": target}, parallel_state=parallel)
                torch.testing.assert_close(target.weight.to_local(), source.weight.to_local(), rtol=0, atol=0)

        # Unversioned chunks may be indistinguishable from a misordered
        # composite TP+EP source, so require independent source evidence.
        path = directory + "legacy_rows"
        dcp.save({"model": {"weight": wrap(1)}}, checkpoint_id=path)
        for target_dim in (0, 1):
            target = {"model": {"weight": wrap(target_dim, empty=True)}}
            with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
                dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
            assert target["model"]["weight"].to_local().count_nonzero() == 0
            dcp.load(
                target,
                checkpoint_id=path,
                planner=_ModelStrictLoadPlanner(strict_model=True, legacy_expert_source_layout="veomni_tp1_ep0_fsdp1"),
            )
            torch.testing.assert_close(target["model"]["weight"].full_tensor(), global_weight, rtol=0, atol=0)

        # The public model-only load/export paths accept the same independently
        # verified declaration; unknown source and optimizer restore stay closed.
        path = directory + "known_legacy_model_only"
        bf16_bias = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
        dcp.save({"model": {"layer.experts.weight": wrap(1), "layer.bias": bf16_bias}}, checkpoint_id=path)
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer.experts = torch.nn.Module()
        model.layer.experts.weight = torch.nn.Parameter(torch.zeros_like(global_weight))
        parallel = SimpleNamespace(dp_mode="ddp")
        with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
            DistributedCheckpointer.load(path, {"model": model}, parallel_state=parallel)
        assert model.layer.experts.weight.count_nonzero() == 0
        with pytest.raises(ValueError, match="only supported for model-only loads"):
            DistributedCheckpointer.load(
                path,
                {"model": model, "optimizer": torch.optim.AdamW(model.parameters())},
                parallel_state=parallel,
                legacy_expert_source_layout="veomni_tp1_ep0_fsdp1",
            )
        assert model.layer.experts.weight.count_nonzero() == 0
        DistributedCheckpointer.load(
            path,
            {"model": model},
            parallel_state=parallel,
            legacy_expert_source_layout="veomni_tp1_ep0_fsdp1",
        )
        torch.testing.assert_close(model.layer.experts.weight, global_weight, rtol=0, atol=0)
        if rank == 0:
            import subprocess
            import sys

            from safetensors.torch import load_file

            from veomni.checkpoint import ckpt_to_state_dict

            with pytest.raises(RuntimeError, match="Cannot determine source expert order"):
                dcp_to_torch_state_dict(path)
            exported = dcp_to_torch_state_dict(path, legacy_expert_source_layout="veomni_tp1_ep0_fsdp1")
            torch.testing.assert_close(exported["layer.experts.weight"], global_weight, rtol=0, atol=0)
            exported = ckpt_to_state_dict(path, legacy_expert_source_layout="veomni_tp1_ep0_fsdp1")
            torch.testing.assert_close(exported["layer.experts.weight"], global_weight, rtol=0, atol=0)
            torch.testing.assert_close(exported["layer.bias"], bf16_bias, rtol=0, atol=0)
            assert exported["layer.experts.weight"].dtype == torch.float32
            assert exported["layer.bias"].dtype == torch.bfloat16
            cli_dir = directory + "known_legacy_hf"
            repo_root = Path(__file__).resolve().parents[2]
            cli_env = os.environ.copy()
            cli_env["PYTHONPATH"] = os.pathsep.join((str(repo_root), cli_env.get("PYTHONPATH", "")))
            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "merge_dcp_to_hf.py"),
                    "--load-dir",
                    path,
                    "--save-dir",
                    cli_dir,
                    "--mode",
                    "full",
                    "--legacy-expert-source-layout",
                    "veomni_tp1_ep0_fsdp1",
                    "--save-dtype",
                    "preserve",
                ],
                check=True,
                timeout=60,
                cwd=repo_root,
                env=cli_env,
            )
            exported = load_file(cli_dir + "/model.safetensors")
            torch.testing.assert_close(exported["layer.experts.weight"], global_weight, rtol=0, atol=0)
            torch.testing.assert_close(exported["layer.bias"], bf16_bias, rtol=0, atol=0)
            assert exported["layer.experts.weight"].dtype == torch.float32
            assert exported["layer.bias"].dtype == torch.bfloat16
        dist.barrier()

        # Legacy metadata used two ordinary Shards, recording the wrong order.
        local = ep_weight.chunk(2, 0)[fsdp_rank].contiguous()
        placements = [Shard(0), Shard(0)] if world == 4 else [Replicate(), Shard(0), Shard(0)]
        legacy = DTensor.from_local(local, mesh, placements)
        for key, strict in [("model", True), ("model", False), ("optimizer", False)]:
            path = directory + key + str(strict)
            dcp.save({key: {"weight": legacy}}, checkpoint_id=path)
            for target_dim in (0, 1):
                target = {key: {"weight": wrap(target_dim, empty=True)}}
                with pytest.raises(dcp.CheckpointException, match="unversioned checkpoint"):
                    dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=strict))
                assert target[key]["weight"].to_local().count_nonzero() == 0

        # Offline export targets plain CPU tensors: an unversioned stacked
        # expert saved only in same-axis chunks cannot prove expert order.
        path = directory + "legacy_ambiguous_export"
        dcp.save({"model": {"layer.experts.weight": legacy}}, checkpoint_id=path)
        if rank == 0:
            with pytest.raises(RuntimeError, match="Cannot determine source expert order"):
                _get_sharding_plan(path)
            with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
                _process_shard({"layer.experts.weight": "model.layer.experts.weight"}, path)
        dist.barrier()

        # A one-dimensional expert bias may also be EP-sharded. Exporting it
        # as a plain tensor must not bypass the same unversioned-source guard.
        bias = torch.arange(8, dtype=torch.float32)
        local_bias = bias.chunk(2, 0)[ep_rank].chunk(2, 0)[fsdp_rank].contiguous()
        placements = [Shard(0), Shard(0)] if world == 4 else [Replicate(), Shard(0), Shard(0)]
        legacy_bias = DTensor.from_local(local_bias, mesh, placements)
        path = directory + "legacy_ambiguous_bias_export"
        dcp.save({"model": {"layer.experts.bias": legacy_bias}}, checkpoint_id=path)
        if rank == 0:
            with pytest.raises(RuntimeError, match="Cannot determine source expert order"):
                dcp_to_torch_state_dict(path)
        dist.barrier()

        # When the source layout is known, the guard also works for a plain
        # tensor target. Never bless legacy ordinary Shard(0)/Shard(0) saves.
        path = directory + "marked_legacy"
        dcp.save({"model": {"weight": legacy}}, checkpoint_id=path, planner=_ExtraParallelSavePlanner())
        for destination in (wrap(0, empty=True), wrap(1, empty=True), torch.zeros_like(global_weight)):
            for declaration in (None, "veomni_tp1_ep0_fsdp1", "veomni_tp1_no_ep"):
                with pytest.raises(dcp.CheckpointException, match="noncanonical expert order"):
                    dcp.load(
                        {"model": {"weight": destination}},
                        checkpoint_id=path,
                        planner=_ModelStrictLoadPlanner(strict_model=True, legacy_expert_source_layout=declaration),
                    )
                local = destination.to_local() if isinstance(destination, DTensor) else destination
                assert local.count_nonzero() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4, 8])
def test_expert_shard_layout_roundtrip_and_legacy_rejection(world, tmp_path):
    mp.spawn(
        _layout_worker, args=(world, str(tmp_path / "dcp"), f"file://{tmp_path / 'rdzv'}"), nprocs=world, join=True
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_unsupported_legacy_expert_source_layout_is_rejected():
    with pytest.raises(ValueError, match="Unsupported legacy expert source layout"):
        _ModelStrictLoadPlanner(strict_model=True, legacy_expert_source_layout="unverified")


def _non_ep_legacy_expert_bias_worker(rank, root, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        expected = torch.arange(8, dtype=torch.float32)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        source = DTensor.from_local(expected.chunk(2)[rank].contiguous(), mesh, [Shard(0)])
        # Emulate a legacy checkpoint written before per-tensor provenance existed.
        dcp.save({"model": {"layer.experts.bias": source}}, checkpoint_id=root, planner=DefaultSavePlanner())
        target = {"model": {"layer.experts.bias": torch.zeros_like(expected)}}
        with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
            dcp.load(target, checkpoint_id=root, planner=_ModelStrictLoadPlanner(strict_model=True))
        assert target["model"]["layer.experts.bias"].count_nonzero() == 0
        dcp.load(
            target,
            checkpoint_id=root,
            planner=_ModelStrictLoadPlanner(strict_model=True, legacy_expert_source_layout="veomni_tp1_no_ep"),
        )
        torch.testing.assert_close(target["model"]["layer.experts.bias"], expected, rtol=0, atol=0)
        if rank == 0:
            exported = dcp_to_torch_state_dict(root, legacy_expert_source_layout="veomni_tp1_no_ep")
            torch.testing.assert_close(exported["layer.experts.bias"], expected, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_legacy_non_ep_fsdp_expert_bias_requires_source_declaration(tmp_path):
    spawn_with_rendezvous(_non_ep_legacy_expert_bias_worker, args=(str(tmp_path / "dcp"),), nprocs=2)


def _unversioned_wrong_target_expert_axis_worker(rank, root, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp, ep = mesh.get_coordinate()
        expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        local = expected.chunk(2, 0)[ep].chunk(2, 0)[fsdp].contiguous()
        source = DTensor.from_local(local, mesh, [Shard(0), Shard(0)])
        dcp.save({"model": {"layer.experts.weight": source}}, checkpoint_id=root)
        target = DTensor.from_local(torch.zeros(4, 4), mesh, [Shard(0), Shard(1)])
        with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
            dcp.load(
                {"model": {"layer.experts.weight": target}},
                checkpoint_id=root,
                planner=_ModelStrictLoadPlanner(strict_model=True),
            )
        assert target.to_local().count_nonzero() == 0
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_unversioned_source_axis_is_not_inferred_from_target(tmp_path):
    spawn_with_rendezvous(_unversioned_wrong_target_expert_axis_worker, args=(str(tmp_path / "dcp"),), nprocs=4)


def _partial_expert_layout_metadata_worker(rank, root, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        unnamed_mesh = DeviceMesh("cpu", mesh.mesh, mesh_dim_names=("axis0", "axis1"))
        fsdp, ep = mesh.get_coordinate()
        expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        good_local = expected.chunk(2, 0)[ep].chunk(2, 1)[fsdp].contiguous()
        bad_local = expected.chunk(2, 0)[ep].chunk(2, 0)[fsdp].contiguous()
        good = DTensor.from_local(good_local, mesh, [Shard(1), Shard(0)])
        bad = DTensor.from_local(bad_local, unnamed_mesh, [Shard(0), Shard(0)])
        source = {"model": {"layer.experts.good": good, "layer.experts.bad": bad}}

        class OldPartialPlanner(_ExtraParallelSavePlanner):
            def create_global_plan(self, all_plans):
                plans, metadata = super().create_global_plan(all_plans)
                layouts = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
                metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"] = {
                    key: value for key, value in layouts.items() if value["expert_order"] == "canonical"
                }
                return plans, metadata

        dcp.save(source, checkpoint_id=root, planner=OldPartialPlanner())
        recorded = dcp.FileSystemReader(root).read_metadata().planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
        assert "model.layer.experts.good" in recorded and "model.layer.experts.bad" not in recorded
        target = {
            "model": {
                f"layer.experts.{name}": DTensor.from_local(torch.zeros(4, 4), mesh, [Shard(1), Shard(0)])
                for name in ("good", "bad")
            }
        }
        with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
            dcp.load(target, checkpoint_id=root, planner=_ModelStrictLoadPlanner(strict_model=True))
        assert all(value.to_local().count_nonzero() == 0 for value in target["model"].values())
        if rank == 0:
            with pytest.raises(RuntimeError, match="Cannot determine source expert order"):
                dcp_to_torch_state_dict(root)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_partial_source_layout_metadata_does_not_authorize_other_experts(tmp_path):
    spawn_with_rendezvous(_partial_expert_layout_metadata_worker, args=(str(tmp_path / "dcp"),), nprocs=4)


def _unversioned_composite_expert_worker(rank, root, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=8)
    try:
        from torch.distributed.tensor.placement_types import _StridedShard

        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("tp", "ep_fsdp", "ep"))
        tp, fsdp, ep = mesh.get_coordinate()
        expected = torch.arange(8 * 16 * 4, dtype=torch.float32).reshape(8, 16, 4)
        local = expected.chunk(2, 0)[ep].chunk(2, 0)[fsdp].chunk(2, 1)[tp].contiguous()
        old = DTensor.from_local(local, mesh, [Shard(1), Shard(0), Shard(0)])
        dcp.save({"model": {"layer.experts.weight": old}}, checkpoint_id=root)
        target = DTensor.from_local(
            torch.zeros_like(local), mesh, [Shard(1), _StridedShard(0, split_factor=2), Shard(0)]
        )
        with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
            dcp.load(
                {"model": {"layer.experts.weight": target}}, checkpoint_id=root, planner=_ModelStrictLoadPlanner(True)
            )
        assert target.to_local().count_nonzero() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=8))
def test_unversioned_composite_expert_layout_is_rejected(tmp_path):
    spawn_with_rendezvous(_unversioned_composite_expert_worker, args=(str(tmp_path / "checkpoint"),), nprocs=8)


def _lora_model(config=None):
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel

    model = torch.nn.Module()
    experts = model.experts = torch.nn.Module()
    experts.num_experts, experts.hidden_dim, experts.intermediate_dim = 8, 128, 256
    experts.act_fn = torch.nn.SiLU()
    experts.gate_up_proj = torch.nn.Parameter(torch.zeros(8, 512, 128))
    experts.down_proj = torch.nn.Parameter(torch.zeros(8, 128, 256))
    config = config or VeOmniLoraConfig.from_yaml(
        {"rank": 8, "alpha": 16, "target_parameters": ["experts.gate_up_proj", "experts.down_proj"]}
    )
    return VeOmniLoraModel(model, config)


@pytest.mark.parametrize("root", ["model", "optimizer"])
def test_versioned_owner_lookup_requires_exact_logical_fqn(root):
    from veomni.checkpoint.dcp_checkpointer import _key_belongs_to_fqns, _logical_extra_parallel_key

    marker = _extra_parallel_layout_key({})
    state_dict = {f"{root}.{marker}": "ep-outer-v1"}
    fqn = "layers.0.experts.gate_proj.lora_A.weight"
    tail = fqn if root == "model" else f"state.{fqn}.exp_avg"
    key = _logical_extra_parallel_key(f"{root}.{marker}__{tail}", state_dict)
    assert _key_belongs_to_fqns(key, {fqn})
    assert not _key_belongs_to_fqns(key, {f"other.{fqn}"})
    assert _logical_extra_parallel_key(f"{root}.{marker}", state_dict) is None
    nested = _logical_extra_parallel_key(f"{root}.{marker}__{marker}__{tail}", state_dict)
    assert not _key_belongs_to_fqns(nested, {fqn})
    unversioned = f"{root}.{marker}__{tail}"
    assert _logical_extra_parallel_key(unversioned, {}) == unversioned
    with pytest.raises(RuntimeError, match="mixes versioned and legacy"):
        _logical_extra_parallel_key(f"{root}.{tail}", state_dict)


@pytest.mark.parametrize("fqn", ["experts.weight", "layers.0.experts.weight"])
def test_versioned_experts_without_proven_owner_are_recorded_as_unproven(fqn):
    marker = _extra_parallel_layout_key({})
    key = f"model.{marker}__{fqn}"
    planner = _ExtraParallelSavePlanner()
    planner.set_up_planner(
        {"model": {marker: "ep-outer-v1", f"{marker}__{fqn}": torch.ones(2, 2)}}, is_coordinator=True
    )
    plan = planner.create_local_plan()
    assert plan.planner_data[_EXPERT_LAYOUTS_KEY][key] == {"expert_order": "unproven"}


@pytest.mark.parametrize("fqn", ["experts.weight", "layers.0.experts.weight"])
@pytest.mark.parametrize("layout_metadata", [False, True])
def test_v2_checkpoint_missing_expert_proof_rejects_load_and_export(tmp_path, fqn, layout_metadata):
    class MissingLayoutPlanner(_ExtraParallelSavePlanner):
        def create_local_plan(self):
            plan = super().create_local_plan()
            plan.planner_data[_EXPERT_LAYOUTS_KEY].clear()
            return plan

    marker = _extra_parallel_layout_key({})
    name = f"{marker}__{fqn}"
    state = {"model": {marker: "ep-outer-v1", name: torch.ones(2, 2)}}
    planner = MissingLayoutPlanner() if layout_metadata else DefaultSavePlanner()
    dcp.save(state, checkpoint_id=tmp_path, planner=planner)
    target = {"model": {marker: "", name: torch.zeros(2, 2)}}
    with pytest.raises(dcp.CheckpointException, match="metadata omits its layout"):
        dcp.load(target, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(strict_model=True))
    assert target["model"][name].count_nonzero() == 0
    with pytest.raises(RuntimeError, match="metadata omits its layout"):
        dcp_to_torch_state_dict(tmp_path)


def test_existing_v2_unproven_expert_checkpoint_rejects_load_and_export(tmp_path):
    """Historical invalid checkpoints remain unreadable after save-side checks."""
    marker = _extra_parallel_layout_key({})
    name = f"{marker}__layers.0.experts.weight"

    class HistoricalUnprovenPlanner(DefaultSavePlanner):
        def create_global_plan(self, plans):
            global_plans, metadata = super().create_global_plan(plans)
            metadata.planner_data[_EXPERT_LAYOUTS_KEY] = {
                "version": 1,
                "tensors": {f"model.{name}": {"expert_order": "unproven"}},
            }
            return global_plans, metadata

    dcp.save(
        {"model": {marker: "ep-outer-v1", name: torch.ones(2, 2)}},
        checkpoint_id=tmp_path,
        planner=HistoricalUnprovenPlanner(),
    )
    target = {"model": {marker: "", name: torch.zeros(2, 2)}}
    with pytest.raises(dcp.CheckpointException, match="noncanonical expert order"):
        dcp.load(target, checkpoint_id=tmp_path, planner=_ModelStrictLoadPlanner(strict_model=True))
    assert target["model"][name].count_nonzero() == 0
    with pytest.raises(RuntimeError, match="noncanonical expert order"):
        dcp_to_torch_state_dict(tmp_path)


def _lora_export_worker(rank, world, shard_dim, directory, ep_dim, canonical, endpoint):
    from veomni.distributed.parallel_plan import SpecInfo
    from veomni.lora import VeOmniLoraConfig
    from veomni.lora.state_dict import get_lora_state_dict, load_adapter_state_dict, strip_adapter_name
    from veomni.lora.weight_loading import load_lora_weights
    from veomni.utils.save_safetensor_utils import save_lora_adapter_with_dcp

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        fsdp_rank, ep_rank = mesh.get_coordinate()
        model = _lora_model()
        model._fqn2spec_info = {}
        expected = {}
        for name, param in list(model.named_parameters()):
            if ".lora_" not in name:
                continue
            # Distinct expert/adapter values reveal incorrect shape and shard order.
            value = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) / 256 + len(expected)
            expected[strip_adapter_name(name, "default")] = value.to(torch.bfloat16)
            local = value.chunk(2, 0)[ep_rank].chunk(2, shard_dim)[fsdp_rank].contiguous()
            sharded = DTensor.from_local(local, mesh["ep_fsdp"], [Shard(shard_dim)])
            module_name, param_name = name.rsplit(".", 1)
            setattr(model.get_submodule(module_name), param_name, torch.nn.Parameter(sharded))
            model._fqn2spec_info[name] = SpecInfo(
                "ep",
                Shard(ep_dim),
                name,
                mesh,
                fsdp_shard_dim=shard_dim,
                global_shape=tuple(value.shape),
                checkpoint_layout_v2=canonical,
            )

        save_lora_adapter_with_dcp(model, directory)
        exported = load_adapter_state_dict(directory)
        assert exported.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(exported[name], value, rtol=0, atol=0)

        # Reload through the public adapter loader, including the saved config.
        reloaded = _lora_model(VeOmniLoraConfig.from_pretrained(directory))
        load_lora_weights(reloaded, directory, init_device="cpu")
        for name, value in get_lora_state_dict(reloaded).items():
            torch.testing.assert_close(value, expected[name].to(value.dtype), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4])
@pytest.mark.parametrize("shard_dim,ep_dim", [(0, 0), (1, 0), (0, -3)])
@pytest.mark.parametrize("canonical", [False, True])
def test_lora_adapter_export_preserves_expert_shard_layout(world, shard_dim, ep_dim, canonical, tmp_path):
    spawn_with_rendezvous(
        _lora_export_worker, args=(world, shard_dim, str(tmp_path / "adapter"), ep_dim, canonical), nprocs=world
    )


def _ordinary_fsdp_expert_export_worker(rank, root, ndim, endpoint):
    from torch.distributed.tensor import distribute_tensor

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        parallel = get_parallel_state()
        mesh = parallel.fsdp_mesh
        shape = (8,) + (4,) * (ndim - 1)
        expected = torch.arange(8 * 4 ** (ndim - 1), dtype=torch.float32).reshape(shape)
        weight = distribute_tensor(expected, mesh, [Shard(0)])
        dcp.save(
            {"model": {"experts": {"weight": weight}}},
            checkpoint_id=root,
            planner=_ExtraParallelSavePlanner(parallel_state=parallel),
        )
        dist.barrier()
        result = [None]
        if rank == 0:
            try:
                metadata = dcp.FileSystemReader(root).read_metadata()
                record = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]["model.experts.weight"]
                assert record["source_kind"] == "veomni_tp1_no_ep"
                exported = dcp_to_torch_state_dict(root)
                torch.testing.assert_close(exported["experts.weight"], expected, rtol=0, atol=0)
            except Exception as exc:
                result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        assert result[0] is None, result[0]

        # A full plain tensor is safe only when the actual runtime proves no
        # ExtraParallel source; without that proof it could be EP-local.
        for proven in (True, False):
            path = f"{root}-plain-{proven}"
            planner = _ExtraParallelSavePlanner(parallel_state=parallel if proven else None)
            dcp.save({"model": {"experts": {"weight": expected}}}, checkpoint_id=path, planner=planner)
            result = [None]
            if rank == 0:
                try:
                    metadata = dcp.FileSystemReader(path).read_metadata()
                    record = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]["model.experts.weight"]
                    assert record["expert_order"] == ("canonical" if proven else "unproven")
                    if proven:
                        torch.testing.assert_close(
                            dcp_to_torch_state_dict(path)["experts.weight"], expected, rtol=0, atol=0
                        )
                    else:
                        with pytest.raises(RuntimeError, match="noncanonical expert order"):
                            dcp_to_torch_state_dict(path)
                except Exception as exc:
                    result[0] = repr(exc)
            dist.broadcast_object_list(result, src=0)
            assert result[0] is None, result[0]

        # The same FSDP-looking tensor is not proof that an earlier EP axis
        # was never stripped; an unknown source must not be certified.
        unproven = root + "-unproven"
        dcp.save(
            {"model": {"experts": {"weight": weight}}}, checkpoint_id=unproven, planner=_ExtraParallelSavePlanner()
        )
        result = [None]
        if rank == 0:
            try:
                metadata = dcp.FileSystemReader(unproven).read_metadata()
                record = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]["model.experts.weight"]
                assert record["expert_order"] == "unproven"
                with pytest.raises(RuntimeError, match="noncanonical expert order"):
                    dcp_to_torch_state_dict(unproven)
            except Exception as exc:
                result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        assert result[0] is None, result[0]

        from unittest.mock import patch

        from veomni.utils.save_safetensor_utils import save_hf_safetensor

        with (
            patch("veomni.utils.save_safetensor_utils.should_use_distributed_hf_safetensor_save", return_value=False),
            pytest.raises(RuntimeError, match="noncanonical expert order|failed on another rank"),
        ):
            save_hf_safetensor(
                save_hf_safetensor_path=f"{root}-hf",
                save_checkpoint_path=unproven,
                ckpt_manager="dcp",
                is_rank_0=rank == 0,
            )
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2

        # Earlier planner metadata did not prove that non-EP dimensions were
        # absent, even when this particular tensor used ordinary FSDP.
        old_v1 = root + "-old-v1"
        dcp.save(
            {"model": {"experts": {"weight": weight}}}, checkpoint_id=old_v1, planner=_LegacyNamedExtraSavePlanner()
        )
        result = [None]
        if rank == 0:
            try:
                with pytest.raises(RuntimeError, match="Cannot determine source expert order"):
                    dcp_to_torch_state_dict(old_v1)
            except Exception as exc:
                result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        assert result[0] is None, result[0]

        # An old expert-only chunk has no provenance, even when the target is
        # ordinary FSDP. It must not be loaded and re-saved as canonical.
        legacy = root + "-legacy"
        dcp.save({"model": {"experts": {"weight": weight}}}, checkpoint_id=legacy)
        target = {"model": {"experts": {"weight": torch.zeros_like(weight)}}}
        with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
            dcp.load(target, checkpoint_id=legacy, planner=_ModelStrictLoadPlanner(strict_model=True))
        assert target["model"]["experts"]["weight"].to_local().count_nonzero() == 0
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [2])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_recorded_non_ep_fsdp_expert_export(world, ndim, tmp_path):
    assert world == 2
    spawn_with_rendezvous(_ordinary_fsdp_expert_export_worker, args=(str(tmp_path / "dcp"), ndim), nprocs=world)


@pytest.mark.parametrize("outer", [Shard(1), Replicate()])
def test_unexplained_mesh_axis_overlap_is_not_certified(outer):
    mesh = DeviceMesh(
        "cpu",
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        mesh_dim_names=("x", "ep_fsdp", "ep"),
        _init_backend=False,
        _rank=0,
    )
    value = DTensor.from_local(torch.zeros(2, 2), mesh, [outer, Shard(1), Shard(0)], run_check=False)
    assert _expert_layout(value)["expert_order"] == ("unproven" if isinstance(outer, Shard) else "canonical")


@pytest.mark.parametrize(
    "placements",
    [
        [Shard(1), _StridedShard(1, split_factor=2), Shard(0)],
        [Replicate(), _StridedShard(0, split_factor=3), Shard(0)],
        [Replicate(), Shard(1), _StridedShard(0, split_factor=2)],
    ],
)
def test_unproven_strided_extra_layout_is_not_certified(placements):
    mesh = DeviceMesh(
        "cpu",
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        mesh_dim_names=("x", "ep_fsdp", "ep"),
        _init_backend=False,
        _rank=0,
    )
    value = DTensor.from_local(torch.zeros(3, 4), mesh, placements, run_check=False)
    assert _expert_layout(value)["expert_order"] == "unproven"


def test_strided_unidentified_mesh_is_not_certified():
    from veomni.checkpoint.dcp_checkpointer import _unidentified_same_axis_shards

    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("x", "y"), _init_backend=False, _rank=0)
    value = DTensor.from_local(
        torch.zeros(2, 2), mesh, [_StridedShard(0, split_factor=2), Replicate()], run_check=False
    )
    assert _unidentified_same_axis_shards(value)


def test_unexplained_strided_axis_does_not_short_circuit_provenance():
    mesh = DeviceMesh(
        "cpu",
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        mesh_dim_names=("x", "ep_fsdp", "ep"),
        _init_backend=False,
        _rank=0,
    )
    value = DTensor.from_local(
        torch.zeros(3, 4, 5), mesh, [_StridedShard(1, split_factor=2), Shard(2), Shard(0)], run_check=False
    )
    assert _expert_layout(value)["expert_order"] == "unproven"


def test_partial_placement_does_not_certify_source_values():
    mesh = DeviceMesh(
        "cpu",
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        mesh_dim_names=("x", "ep_fsdp", "ep"),
        _init_backend=False,
        _rank=0,
    )
    value = DTensor.from_local(torch.zeros(3, 4), mesh, [Partial(), Shard(1), Shard(0)], run_check=False)
    assert _expert_layout(value)["expert_order"] == "unproven"


def _ple_and_ordinary_expert_worker(rank, root, endpoint):
    from torch.distributed.tensor import distribute_tensor

    from veomni.distributed.parallel_plan import ParallelPlan
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            device_type="cpu",
            extra_parallel_names=("ep", "ple"),
            extra_parallel_sizes=(1, 2),
            extra_parallel_placement_innermost=(False, False),
        )
        parallel = get_parallel_state()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = torch.nn.Module()
                self.experts.weight = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(8, 4))
                self.experts.alias = self.experts.weight
                self.ple = torch.nn.Embedding(8, 4)
                self.shard_expert_alias = False

            def get_parallel_plan(self):
                entries = {"ple.weight": Shard(0)}
                if self.shard_expert_alias:
                    entries["experts.alias"] = Shard(0)
                return ParallelPlan({"ple": entries})

        model = Model()
        model._fqn2spec_info = model.get_parallel_plan().apply(model, parallel.extra_parallel_fsdp_device_mesh)
        expected = model.experts.weight.detach().clone()
        sharded = torch.nn.Parameter(distribute_tensor(expected, parallel.fsdp_mesh, [Shard(0)]))
        model.experts.weight = sharded
        model.experts.alias = sharded
        for conflicting_plan in (False, True):
            model.shard_expert_alias = conflicting_plan
            path = f"{root}-{conflicting_plan}"
            planner = _ExtraParallelSavePlanner(parallel_state=parallel, source_model=model)
            dcp.save({"model": {"experts": {"weight": sharded}}}, checkpoint_id=path, planner=planner)
            result = [None]
            if rank == 0:
                try:
                    metadata = dcp.FileSystemReader(path).read_metadata()
                    record = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]["model.experts.weight"]
                    assert record["expert_order"] == ("unproven" if conflicting_plan else "canonical")
                    if conflicting_plan:
                        with pytest.raises(RuntimeError, match="noncanonical expert order"):
                            dcp_to_torch_state_dict(path)
                    else:
                        torch.testing.assert_close(
                            dcp_to_torch_state_dict(path)["experts.weight"], expected, rtol=0, atol=0
                        )
                except Exception as exc:
                    result[0] = repr(exc)
            dist.broadcast_object_list(result, src=0)
            assert result[0] is None, result[0]
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("model.layers.0.experts.weight", True),
        ("optimizer.state.layers.0.experts.weight.exp_avg", True),
        ("optimizer.state.layers.0.experts.weight.exp_avg.extra", True),
        ("optimizer.state.layers.0.experts.weight_suffix.exp_avg", False),
        ("model.layers.0.experts.weight_suffix", False),
    ],
)
def test_expert_source_key_matches_parameter_boundaries(key, expected):
    from veomni.checkpoint.dcp_checkpointer import _key_belongs_to_fqns

    assert _key_belongs_to_fqns(key, {"layers.0.experts.weight"}) is expected


def _single_rank_uninitialized_expert_worker(rank, root, endpoint):
    from torch.distributed.tensor import distribute_tensor

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import clear_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=1, timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))

        def build():
            model = torch.nn.Module()
            model.experts = torch.nn.Module()
            model.experts.weight = torch.nn.Parameter(
                distribute_tensor(torch.arange(8, dtype=torch.float32), mesh, [Shard(0)])
            )
            return model

        model = build()
        DistributedCheckpointer.save(root, {"model": model})
        target = build()
        target.experts.weight.data.zero_()
        DistributedCheckpointer.load(root, {"model": target})
        torch.testing.assert_close(target.experts.weight.to_local(), model.experts.weight.to_local(), rtol=0, atol=0)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [1])
def test_single_rank_without_parallel_state_retains_expert_checkpoint(world, tmp_path):
    spawn_with_rendezvous(_single_rank_uninitialized_expert_worker, args=(str(tmp_path / "dcp"),), nprocs=world)


def test_no_distributed_initialization_retains_expert_checkpoint(tmp_path, monkeypatch):
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import clear_parallel_state

    monkeypatch.setenv("WORLD_SIZE", "1")
    clear_parallel_state()
    assert not dist.is_initialized()
    model = torch.nn.Module()
    model.experts = torch.nn.Linear(2, 2)
    expected = model.experts.weight.detach().clone()
    DistributedCheckpointer.save(str(tmp_path), {"model": model})
    with torch.no_grad():
        model.experts.weight.zero_()
    DistributedCheckpointer.load(str(tmp_path), {"model": model})
    torch.testing.assert_close(model.experts.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shared", [False, True])
def test_single_process_lora_export_without_parallel_state(tmp_path, monkeypatch, shared):
    from veomni.distributed.parallel_state import clear_parallel_state
    from veomni.lora import VeOmniLoraConfig
    from veomni.lora.state_dict import get_lora_state_dict, load_adapter_state_dict
    from veomni.utils.save_safetensor_utils import save_lora_adapter_with_dcp

    monkeypatch.setenv("WORLD_SIZE", "1")
    clear_parallel_state()
    assert not dist.is_initialized()
    config = VeOmniLoraConfig.from_yaml(
        {
            "rank": 8,
            "alpha": 16,
            "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
            "share_expert_lora": shared,
        }
    )
    model = _lora_model(config)
    expected = {
        key: value.detach().to(torch.bfloat16).cpu()
        for key, value in get_lora_state_dict(model, config=model.get_lora_config()).items()
    }
    save_lora_adapter_with_dcp(model, str(tmp_path))
    actual = load_adapter_state_dict(str(tmp_path))
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [2])
def test_ple_with_ep1_proves_ordinary_expert_from_plan(world, tmp_path):
    spawn_with_rendezvous(_ple_and_ordinary_expert_worker, args=(str(tmp_path / "dcp"),), nprocs=world)


def _ple_ep1_lora_worker(rank, root, unproven, endpoint):
    from torch.distributed.tensor import distribute_tensor

    from veomni.distributed.parallel_plan import ParallelPlan, get_runtime_parallel_plan
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state
    from veomni.lora import VeOmniLoraConfig
    from veomni.lora.state_dict import get_lora_state_dict, load_adapter_state_dict, strip_adapter_name
    from veomni.utils.save_safetensor_utils import save_lora_adapter_with_dcp

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            device_type="cpu",
            extra_parallel_names=("ep", "ple"),
            extra_parallel_sizes=(1, 2),
            extra_parallel_placement_innermost=(False, False),
        )
        parallel = get_parallel_state()
        torch.manual_seed(42)
        config = VeOmniLoraConfig.from_yaml(
            {"rank": 8, "alpha": 16, "target_parameters": ["experts.gate_up_proj", "experts.down_proj"], "bias": "all"}
        )
        model = _lora_model(config)
        base = model.base_model.model
        base.experts.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        base.ple = torch.nn.Embedding(8, 4)
        base.get_parallel_plan = lambda: ParallelPlan({"ple": {"ple.weight": Shard(0)}})
        plan = get_runtime_parallel_plan(model)
        model._fqn2spec_info = plan.apply(model, parallel.extra_parallel_fsdp_device_mesh)
        selected = get_lora_state_dict(model, config=model.get_lora_config())
        assert any("experts.bias" in name for name in selected)
        for name, param in list(model.named_parameters()):
            if strip_adapter_name(name, "default") not in selected:
                continue
            parent, attr = name.rsplit(".", 1)
            setattr(
                model.get_submodule(parent),
                attr,
                torch.nn.Parameter(distribute_tensor(param.detach(), parallel.fsdp_mesh, [Shard(0)])),
            )
        selected = get_lora_state_dict(model, config=model.get_lora_config())
        expected = {
            name: (value.full_tensor() if isinstance(value, DTensor) else value).detach().to(torch.bfloat16).cpu()
            for name, value in selected.items()
        }
        if unproven:
            base.get_parallel_plan = lambda: None
            with pytest.raises(RuntimeError, match="noncanonical expert order|failed on another rank"):
                save_lora_adapter_with_dcp(model, root, parallel_state=parallel)
            completed = torch.tensor(1)
            dist.all_reduce(completed)
            assert completed.item() == 2
            return
        save_lora_adapter_with_dcp(model, root, parallel_state=parallel)
        exported = load_adapter_state_dict(root)
        assert exported.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [2])
@pytest.mark.parametrize("unproven", [False, True])
def test_ple_ep1_lora_bias_all_checkpoint_export(world, unproven, tmp_path):
    spawn_with_rendezvous(_ple_ep1_lora_worker, args=(str(tmp_path / "adapter"), unproven), nprocs=world)


def _ep_shared_lora_worker(rank, root, mode, endpoint):
    from torch.distributed.tensor import distribute_tensor

    from veomni.distributed.parallel_plan import ParallelPlan, get_runtime_parallel_plan
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state
    from veomni.lora import VeOmniLoraConfig
    from veomni.lora.moe_layers import LoraSharedExperts
    from veomni.lora.state_dict import get_lora_state_dict, load_adapter_state_dict, strip_adapter_name
    from veomni.utils.save_safetensor_utils import save_lora_adapter_with_dcp

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            device_type="cpu",
            extra_parallel_names=("ep",),
            extra_parallel_sizes=(2,),
            extra_parallel_placement_innermost=(False,),
        )
        parallel = get_parallel_state()
        torch.manual_seed(42)
        config = VeOmniLoraConfig.from_yaml(
            {
                "rank": 8,
                "alpha": 16,
                "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
                "share_expert_lora": True,
                "bias": "all",
            }
        )
        model = _lora_model(config)
        base = model.base_model.model
        assert isinstance(base.experts, LoraSharedExperts)
        if mode == "unproven_bias":
            base.experts.bias = torch.nn.Parameter(torch.ones(8, 8))
        base.get_parallel_plan = lambda: ParallelPlan(
            {"ep": {"experts.gate_up_proj": Shard(0), "experts.down_proj": Shard(0)}}
        )
        plan = get_runtime_parallel_plan(model)
        model._fqn2spec_info = plan.apply(model, parallel.extra_parallel_fsdp_device_mesh)
        selected = get_lora_state_dict(model, config=model.get_lora_config())
        ep_mesh = parallel.extra_parallel_fsdp_device_mesh["ep"]
        fsdp_mesh = ep_mesh[ep_mesh.mesh_dim_names[:-1]]
        for name, param in list(model.named_parameters()):
            if strip_adapter_name(name, "default") not in selected:
                continue
            parent, attr = name.rsplit(".", 1)
            setattr(
                model.get_submodule(parent),
                attr,
                torch.nn.Parameter(distribute_tensor(param.detach(), fsdp_mesh, [Shard(1)])),
            )
        selected = get_lora_state_dict(model, config=model.get_lora_config())
        expected = {name: value.full_tensor().detach().to(torch.bfloat16).cpu() for name, value in selected.items()}
        if mode in ("plan_conflict", "v2_plan_conflict"):
            base.get_parallel_plan = lambda: ParallelPlan(
                {"ep": {"experts.gate_proj.lora_A.default.weight": Shard(0)}}
            )
        if mode.startswith("v2_"):
            from veomni.checkpoint.dcp_checkpointer import _namespace_extra_parallel_layout_state

            marker = _extra_parallel_layout_key({})
            state = {"model": _namespace_extra_parallel_layout_state(selected, marker)}
            planner = _ExtraParallelSavePlanner(parallel_state=parallel, source_model=model, adapter_name="default")
            if mode == "v2_plan_conflict":
                with pytest.raises(dcp.CheckpointException, match="unproven expert layout"):
                    dcp.save(state, checkpoint_id=root, planner=planner)
                assert not (Path(root) / ".metadata").exists()
                completed = torch.tensor(1)
                dist.all_reduce(completed)
                assert completed.item() == 2
                return
            dcp.save(state, checkpoint_id=root, planner=planner)
            metadata = dcp.FileSystemReader(root).read_metadata()
            records = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
            orders = []
            for name in selected:
                if "experts" not in name.split("."):
                    continue
                key = f"model.{marker}__{name}"
                assert key in records
                orders.append(records[key]["expert_order"])
            assert orders
            assert all(o == "canonical" for o in orders)
            target = {
                "model": _namespace_extra_parallel_layout_state(
                    {k: torch.zeros_like(v) for k, v in selected.items()}, marker
                )
            }
            dcp.load(target, checkpoint_id=root, planner=_ModelStrictLoadPlanner(strict_model=True))
            for name, value in selected.items():
                torch.testing.assert_close(
                    target["model"][f"{marker}__{name}"].to_local(), value.to_local(), rtol=0, atol=0
                )
            expected_fp32 = {name: value.full_tensor().detach().cpu() for name, value in selected.items()}
            if rank == 0:
                exported = dcp_to_torch_state_dict(root)
                for name, value in expected_fp32.items():
                    torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
            dist.barrier()
            return
        if mode != "valid":
            with pytest.raises(RuntimeError, match="noncanonical expert order|failed on another rank"):
                save_lora_adapter_with_dcp(model, root, parallel_state=parallel)
            completed = torch.tensor(1)
            dist.all_reduce(completed)
            assert completed.item() == 2
            return
        save_lora_adapter_with_dcp(model, root, parallel_state=parallel)
        exported = load_adapter_state_dict(root)
        assert exported.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [2])
@pytest.mark.parametrize("mode", ["valid", "plan_conflict", "unproven_bias", "v2_valid", "v2_plan_conflict"])
def test_ep_replicated_shared_expert_lora_export(world, mode, tmp_path):
    spawn_with_rendezvous(_ep_shared_lora_worker, args=(str(tmp_path / "adapter"), mode), nprocs=world)


def _unproven_v2_high_level_save_worker(rank, root, endpoint):
    from torch.distributed.tensor import distribute_tensor

    from veomni.checkpoint.layout import weights_dir
    from veomni.distributed.parallel_plan import ParallelPlan
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.Module()
            self.experts.weight = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(8, 4))
            self.experts.bias = torch.nn.Parameter(torch.ones(4, 4))

        def get_parallel_plan(self):
            return ParallelPlan({"ep": {"experts.weight": Shard(0)}})

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            device_type="cpu",
            extra_parallel_names=("ep",),
            extra_parallel_sizes=(2,),
            extra_parallel_placement_innermost=(False,),
        )
        parallel = get_parallel_state()
        model = Model()
        model.get_parallel_plan().apply(model, parallel.extra_parallel_fsdp_device_mesh)
        mesh = parallel.extra_parallel_fsdp_device_mesh["ep"]
        fsdp_mesh = mesh[mesh.mesh_dim_names[:-1]]
        for name, param in list(model.named_parameters()):
            module, attr = name.rsplit(".", 1)
            value = distribute_tensor(param.detach().clone(), fsdp_mesh, [Shard(1)])
            setattr(model.get_submodule(module), attr, torch.nn.Parameter(value))
        model._fqn2spec_info["experts.weight"].checkpoint_layout_v2 = True

        with pytest.raises(dcp.CheckpointException, match="unproven expert layout"):
            DistributedCheckpointer.save(root, {"model": model}, parallel_state=parallel)
        assert not (Path(weights_dir(root)) / ".metadata").exists()
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [2])
def test_unproven_v2_expert_layout_fails_before_checkpoint_publication(world, tmp_path):
    spawn_with_rendezvous(_unproven_v2_high_level_save_worker, args=(str(tmp_path / "checkpoint"),), nprocs=world)


def _pure_extra_parallel_worker(rank, root, name, world, endpoint):
    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world, timeout=timedelta(seconds=30)
    )
    try:
        shape = (2, 2, 2) if world == 8 else (world // 2, 2)
        names = (f"{name}_replicate", f"{name}_fsdp", name) if world == 8 else (f"{name}_fsdp", name)
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
        fsdp_mesh = mesh[names[:-1]]
        expected = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        local = expected.chunk(2, dim=0)[mesh.get_local_rank(name)].contiguous()
        restored_axis = restore_extra_parallel_dim(local, mesh, fsdp_mesh, ep_shard_dim=0)
        torch.testing.assert_close(restored_axis.full_tensor(), expected, rtol=0, atol=0)
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=(name,), extra_parallel_fsdp_device_mesh={name: mesh}
        )

        def model_with(local_value):
            model = torch.nn.Module()
            model.experts = torch.nn.Module()
            model.embedding = torch.nn.Module()
            for module_name in ("experts", "embedding"):
                getattr(model, module_name).weight = torch.nn.Parameter(local_value.clone())
            model._fqn2spec_info = {
                f"{module_name}.weight": SpecInfo(name, Shard(0), f"{module_name}.weight", mesh, 1)
                for module_name in ("experts", "embedding")
            }
            return model

        source = model_with(local)
        planner = _ExtraParallelSavePlanner(parallel_state=parallel)
        dcp.save({"model": ModelState(source, parallel_state=parallel)}, checkpoint_id=root, planner=planner)
        result = [None]
        if rank == 0:
            try:
                exported = dcp_to_torch_state_dict(root)
                for key in ("experts.weight", "embedding.weight"):
                    torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)
            except Exception as exc:
                result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        assert result[0] is None, result[0]
        target = model_with(torch.zeros_like(local))
        dcp.load(
            {"model": ModelState(target, parallel_state=parallel)},
            checkpoint_id=root,
            planner=_ModelStrictLoadPlanner(strict_model=True),
        )
        for module_name in ("experts", "embedding"):
            torch.testing.assert_close(getattr(target, module_name).weight, local, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("name", ["ep", "ple"])
def test_plain_pure_extra_parallel_preserves_global_values(world, name, tmp_path):
    spawn_with_rendezvous(_pure_extra_parallel_worker, args=(str(tmp_path / "dcp"), name, world), nprocs=world)


def _plain_optimizer_unproven_worker(rank, hsdp, endpoint):
    from veomni.checkpoint.dcp_checkpointer import _apply_extra_parallel_dim
    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        shape = (2, 1, 2) if hsdp else (2, 2)
        names = ("ep_replicate", "ep_fsdp", "ep") if hsdp else ("ep_fsdp", "ep")
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
        parallel = SimpleNamespace(extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh})
        spec = SpecInfo("ep", Shard(0), "experts.weight", mesh, 1)
        # Values differ between FSDP ranks and must not be deduplicated as replicas.
        state = {"state.experts.weight.exp_avg": torch.full((4, 4), float(rank))}
        with pytest.raises(RuntimeError, match="no proven replica layout"):
            _apply_extra_parallel_dim(state, {"experts.weight": spec}, parallel, "restore", key_match="substring")
        torch.testing.assert_close(state["state.experts.weight.exp_avg"], torch.full((4, 4), float(rank)))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4])
@pytest.mark.parametrize("hsdp", [False, True])
def test_plain_optimizer_shards_are_not_treated_as_replicas(world, hsdp):
    spawn_with_rendezvous(_plain_optimizer_unproven_worker, args=(hsdp,), nprocs=world)


def _sparse_plain_optimizer_failure_worker(rank, endpoint):
    from veomni.checkpoint.dcp_checkpointer import OptimizerState
    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )
        model = torch.nn.Module()
        model._fqn2spec_info = {"experts.weight": SpecInfo("ep", Shard(0), "experts.weight", mesh, 1)}

        class SparseOptimizer:
            _is_multi_optimizer = True

            def state_dict(self):
                return {"state.experts.weight.exp_avg": torch.ones(4, 4)} if rank == 0 else {}

        state = OptimizerState(model, SparseOptimizer(), parallel_state=parallel)
        with pytest.raises(RuntimeError, match="Raw DCP MultiOptimizer"):
            state.state_dict()
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 4
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4])
def test_raw_multi_optimizer_is_rejected_on_all_ranks(world):
    spawn_with_rendezvous(_sparse_plain_optimizer_failure_worker, args=(), nprocs=world)


def _inconsistent_optimizer_preflight_worker(rank, case, endpoint):
    import veomni.checkpoint.dcp_checkpointer as checkpoint_module
    from veomni.checkpoint.dcp_checkpointer import OptimizerState
    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )
        model = torch.nn.Module()
        model._fqn2spec_info = {"experts.weight": SpecInfo("ep", Shard(0), "experts.weight", mesh, 1)}

        class SparseOptimizer:
            called = False

            @property
            def _is_multi_optimizer(self):
                if case == "type_property_error" and rank == 0:
                    raise ValueError("injected optimizer type lookup error")
                return not (case == "wrong_type_collective" and rank == 0)

            def state_dict(self):
                self.called = True
                dist.barrier()
                return {"state.experts.weight.exp_avg": torch.ones(4, 4)} if rank == 0 else {}

        leaf = SparseOptimizer()
        state = OptimizerState(model, leaf, parallel_state=parallel)
        coordinator = checkpoint_module.raise_if_any_rank_failed
        coordinated = []

        def record_coordination(error, what, group=None):
            coordinated.append(what)
            return coordinator(error, what, group=group)

        checkpoint_module.raise_if_any_rank_failed = record_coordination
        expected_type = ValueError if case == "type_property_error" and rank == 0 else RuntimeError
        expected_message = (
            "injected optimizer type lookup error"
            if expected_type is ValueError
            else "optimizer is not a MultiOptimizer"
            if rank == 0
            else "Raw DCP MultiOptimizer"
        )
        with pytest.raises(expected_type, match=expected_message):
            state.state_dict()
        assert coordinated == ["validating ExtraParallel optimizer checkpoint"]
        assert not leaf.called
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 4
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4])
@pytest.mark.parametrize("case", ["wrong_type_collective", "type_property_error"])
def test_inconsistent_optimizer_preflight_exits_all_ranks(world, case):
    spawn_with_rendezvous(_inconsistent_optimizer_preflight_worker, args=(case,), nprocs=world)


def test_extra_parallel_rejects_non_multi_optimizer_locally():
    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    model = SimpleNamespace(_fqn2spec_info={})
    state = OptimizerState(model, object(), parallel_state=SimpleNamespace(dp_mode="fsdp2"))
    with pytest.raises(RuntimeError, match="optimizer is not a MultiOptimizer"):
        state.state_dict()


def _subgroup_sparse_optimizer_failure_worker(rank, endpoint):
    from veomni.checkpoint.dcp_checkpointer import OptimizerState
    from veomni.distributed.parallel_plan import SpecInfo

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        group = dist.new_group([0, 1], timeout=timedelta(seconds=20))
        mesh = DeviceMesh("cpu", [[0], [1]], mesh_dim_names=("ep_fsdp", "ep"))
        if rank < 2:
            parallel = SimpleNamespace(
                dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
            )
            model = torch.nn.Module()
            model._fqn2spec_info = {"experts.weight": SpecInfo("ep", Shard(0), "experts.weight", mesh, 1)}

            class SparseOptimizer:
                _is_multi_optimizer = True

                def state_dict(self):
                    return {"state.experts.weight.exp_avg": torch.ones(4, 4)} if rank == 0 else {}

            state = OptimizerState(model, SparseOptimizer(), parallel_state=parallel, load=True, process_group=group)
            with pytest.raises(RuntimeError, match="Raw DCP MultiOptimizer"):
                state.state_dict()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4])
def test_raw_multi_optimizer_is_rejected_with_subgroup(world):
    spawn_with_rendezvous(_subgroup_sparse_optimizer_failure_worker, args=(), nprocs=world)


class _LegacyNamedExtraSavePlanner(DefaultSavePlanner):
    def create_global_plan(self, all_plans):
        plans, metadata = super().create_global_plan(all_plans)
        metadata.planner_data = {**(metadata.planner_data or {}), _EXPERT_LAYOUTS_KEY: {"version": 1, "tensors": {}}}
        return plans, metadata


def _named_extra_expert_export_worker(rank, root, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("custom_fsdp", "custom"))
        fsdp_mesh = mesh["custom_fsdp"]
        fsdp_rank, extra_rank = mesh.get_coordinate()
        expected = torch.arange(8 * 16 * 4, dtype=torch.float32).reshape(8, 16, 4)
        local = expected.chunk(2, 0)[extra_rank].chunk(2, 0)[fsdp_rank].contiguous()
        local_dt = DTensor.from_local(local, fsdp_mesh, [Shard(0)])
        canonical = restore_extra_parallel_dim(local_dt, mesh, fsdp_mesh, 0, 0)
        legacy = DTensor.from_local(local, mesh, [Shard(0), Shard(0)])
        key = "model.layers.0.mlp.experts.gate_up_proj"
        for order, source, planner in (
            ("canonical", canonical, _ExtraParallelSavePlanner()),
            ("legacy_fsdp_first", legacy, _ExtraParallelSavePlanner()),
            ("legacy_unrecorded", legacy, _LegacyNamedExtraSavePlanner()),
        ):
            path = f"{root}-{order}"
            dcp.save(
                {
                    "model": {key: source},
                    "optimizer": {"experts": {"moment": source.clone(), "step": torch.tensor(1.0)}},
                },
                checkpoint_id=path,
                planner=planner,
            )
            result = [None]
            if rank == 0:
                try:
                    metadata = dcp.FileSystemReader(path).read_metadata()
                    records = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
                    if order != "legacy_unrecorded":
                        assert records[f"model.{key}"]["expert_order"] == order
                        assert records["optimizer.experts.moment"]["expert_order"] == order
                    if order == "canonical":
                        torch.testing.assert_close(dcp_to_torch_state_dict(path)[key], expected, rtol=0, atol=0)
                    else:
                        with pytest.raises(RuntimeError, match="noncanonical expert order|Cannot determine source"):
                            dcp_to_torch_state_dict(path)
                except Exception as exc:
                    result[0] = repr(exc)
            dist.broadcast_object_list(result, src=0)
            assert result[0] is None, result[0]
            if order == "legacy_unrecorded":
                target = {"model": {key: torch.zeros_like(canonical)}}
                with pytest.raises(dcp.CheckpointException, match="Cannot determine source"):
                    dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
            target = {"optimizer": {"experts": {"moment": torch.zeros_like(canonical), "step": torch.tensor(0.0)}}}
            if order == "canonical":
                dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=False))
                torch.testing.assert_close(
                    target["optimizer"]["experts"]["moment"].full_tensor(), expected, rtol=0, atol=0
                )
                torch.testing.assert_close(target["optimizer"]["experts"]["step"], torch.tensor(1.0), rtol=0, atol=0)
            else:
                with pytest.raises(dcp.CheckpointException, match="noncanonical expert order|Cannot determine source"):
                    dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=False))

        scalar_path = f"{root}-legacy-step"
        dcp.save(
            {"optimizer": {"experts": {"step": torch.tensor(1.0)}}},
            checkpoint_id=scalar_path,
            planner=_LegacyNamedExtraSavePlanner(),
        )
        scalar_target = {"optimizer": {"experts": {"step": torch.tensor(0.0)}}}
        dcp.load(scalar_target, checkpoint_id=scalar_path, planner=_ModelStrictLoadPlanner(strict_model=False))
        torch.testing.assert_close(scalar_target["optimizer"]["experts"]["step"], torch.tensor(1.0), rtol=0, atol=0)

        # The old planner did not recognize ``custom``. A v2 marker describes
        # the EP conversion but does not prove all shared expert replicas, so a
        # missing per-tensor layout must not be silently accepted.
        marker = _extra_parallel_layout_key({})
        canonical_mesh = _extra_parallel_first_mesh(mesh)
        namespaced = DTensor.from_local(
            local,
            canonical_mesh,
            [Shard(0), Shard(0)],
            run_check=False,
            shape=expected.shape,
            stride=expected.stride(),
        )
        path = f"{root}-legacy-v2"
        dcp.save(
            {"model": {marker: "ep-outer-v1", f"{marker}__{key}": namespaced}},
            checkpoint_id=path,
            planner=_LegacyNamedExtraSavePlanner(),
        )
        target = {"model": {marker: "", f"{marker}__{key}": torch.zeros_like(namespaced)}}
        with pytest.raises(dcp.CheckpointException, match="metadata omits its layout"):
            dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
        result = [None]
        if rank == 0:
            try:
                with pytest.raises(RuntimeError, match="metadata omits its layout"):
                    dcp_to_torch_state_dict(path)
            except Exception as exc:
                result[0] = repr(exc)
        dist.broadcast_object_list(result, src=0)
        assert result[0] is None, result[0]

        for names in (("axis_a", "axis_b"), ("fsdp", "ep")):
            unknown_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=names)
            unknown = DTensor.from_local(local, unknown_mesh, [Shard(0), Shard(0)])
            path = f"{root}-unknown-{names[0]}"
            dcp.save({"model": {key: unknown}}, checkpoint_id=path, planner=_ExtraParallelSavePlanner())
            result = [None]
            if rank == 0:
                try:
                    metadata = dcp.FileSystemReader(path).read_metadata()
                    record = metadata.planner_data[_EXPERT_LAYOUTS_KEY]["tensors"][f"model.{key}"]
                    assert record["expert_order"] == "unproven"
                    with pytest.raises(RuntimeError, match="noncanonical expert order"):
                        dcp_to_torch_state_dict(path)
                except Exception as exc:
                    result[0] = repr(exc)
            dist.broadcast_object_list(result, src=0)
            assert result[0] is None, result[0]
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4])
def test_named_extra_parallel_expert_export_checks_source_order(world, tmp_path):
    spawn_with_rendezvous(_named_extra_expert_export_worker, args=(str(tmp_path / "dcp"),), nprocs=world)


def test_export_preflight_preserves_existing_hf_and_allows_filtered_lora(tmp_path):
    import json

    from safetensors.torch import load_file

    from scripts.merge_dcp_to_hf import save_lora_adapter_weights, save_model_weights
    from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan

    old = {"model.aa.weight": torch.full((2, 2), 1.0), "model.zz.experts.weight": torch.full((2, 2), 2.0)}
    new = {"model.aa.weight": torch.full((2, 2), 7.0), "model.zz.experts.weight": torch.full((2, 2), 8.0)}
    old_path, new_path, output = tmp_path / "old", tmp_path / "new", tmp_path / "hf"
    dcp.save(old, checkpoint_id=old_path)
    save_model_weights(str(output), str(old_path), save_dtype=None, shard_size=16)
    index_path = output / "model.safetensors.index.json"
    original_index = index_path.read_bytes()
    mapping = json.loads(original_index)["weight_map"]
    original = {name: load_file(str(output / filename))[name].clone() for name, filename in mapping.items()}

    dcp.save(new, checkpoint_id=new_path, planner=_ExtraParallelSavePlanner())
    with pytest.raises(RuntimeError, match="noncanonical expert order"):
        save_model_weights(str(output), str(new_path), save_dtype=None, shard_size=16)
    assert index_path.read_bytes() == original_index
    for name, filename in mapping.items():
        torch.testing.assert_close(load_file(str(output / filename))[name], original[name], rtol=0, atol=0)
    filtered, _, _ = _get_sharding_plan(new_path, drop_hf_keys={"zz.experts.weight"})
    assert "aa.weight" in filtered and "zz.experts.weight" not in filtered

    lora_key = "model.base_model.model.proj.lora_A.default.weight"
    lora_path, lora_output = tmp_path / "lora-dcp", tmp_path / "lora-hf"
    dcp.save({**new, lora_key: torch.ones(2, 2)}, checkpoint_id=lora_path, planner=_ExtraParallelSavePlanner())
    save_lora_adapter_weights(str(lora_output), str(lora_path), save_dtype=None)
    lora = load_file(str(lora_output / "adapter_model.safetensors"))
    torch.testing.assert_close(lora["base_model.model.proj.lora_A.default.weight"], torch.ones(2, 2), rtol=0, atol=0)
