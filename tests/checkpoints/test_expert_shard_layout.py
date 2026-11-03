# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU/Gloo regressions for canonical EP-first DCP expert coordinates."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from scripts.ci.accelerator.launch_plan import LaunchSpec, from_parameter
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import (
    _EXPERT_LAYOUTS_KEY,
    DistributedCheckpointer,
    _ExtraParallelSavePlanner,
    _get_sharding_plan,
    _ModelStrictLoadPlanner,
    _process_shard,
    dcp_to_torch_state_dict,
    drop_extra_parallel_dim,
    restore_extra_parallel_dim,
)


def _layout_worker(rank, world, directory, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=90))
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
        with pytest.raises(RuntimeError, match="FSDP shard dimension is outside"):
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

        # An unversioned row-sharded source can have identical chunks to a
        # misordered composite TP+EP source. Require source layout evidence.
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

            with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
                dcp_to_torch_state_dict(path)
            exported = dcp_to_torch_state_dict(path, legacy_expert_source_layout="veomni_tp1_ep0_fsdp1")
            torch.testing.assert_close(exported["layer.experts.weight"], global_weight, rtol=0, atol=0)
            exported = ckpt_to_state_dict(path, legacy_expert_source_layout="veomni_tp1_ep0_fsdp1")
            torch.testing.assert_close(exported["layer.experts.weight"], global_weight, rtol=0, atol=0)
            torch.testing.assert_close(exported["layer.bias"], bf16_bias, rtol=0, atol=0)
            assert exported["layer.experts.weight"].dtype == torch.float32
            assert exported["layer.bias"].dtype == torch.bfloat16
            cli_dir = directory + "known_legacy_hf"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/merge_dcp_to_hf.py",
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
            plan, _, _ = _get_sharding_plan(path)
            with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
                _process_shard(plan, path)
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
            with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
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
        dcp.save({"model": {"layer.experts.bias": source}}, checkpoint_id=root, planner=_ExtraParallelSavePlanner())
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
        dcp.save(source, checkpoint_id=root, planner=_ExtraParallelSavePlanner())
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
            with pytest.raises(dcp.CheckpointException, match="Cannot determine source expert order"):
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
