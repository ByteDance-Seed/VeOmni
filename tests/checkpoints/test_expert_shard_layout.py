# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU/Gloo regressions for canonical EP-first DCP expert coordinates."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from scripts.ci.accelerator.launch_plan import from_parameter
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import (
    _EXPERT_LAYOUTS_KEY,
    _ExtraParallelSavePlanner,
    _ModelStrictLoadPlanner,
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

        # Old row-sharded files are identifiable from the saved chunks, and
        # can load into either target dimension without a source version tag.
        path = directory + "legacy_rows"
        dcp.save({"model": {"weight": wrap(1)}}, checkpoint_id=path)
        for target_dim in (0, 1):
            target = {"model": {"weight": wrap(target_dim, empty=True)}}
            dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
            torch.testing.assert_close(target["model"]["weight"].full_tensor(), global_weight, rtol=0, atol=0)

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

        # When the source layout is known, the guard also works for a plain
        # tensor target. Never bless legacy ordinary Shard(0)/Shard(0) saves.
        path = directory + "marked_legacy"
        dcp.save({"model": {"weight": legacy}}, checkpoint_id=path, planner=_ExtraParallelSavePlanner())
        for destination in (wrap(0, empty=True), wrap(1, empty=True), torch.zeros_like(global_weight)):
            with pytest.raises(dcp.CheckpointException, match="noncanonical expert order"):
                dcp.load(
                    {"model": {"weight": destination}},
                    checkpoint_id=path,
                    planner=_ModelStrictLoadPlanner(strict_model=True),
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


def _lora_export_worker(rank, world, shard_dim, directory, endpoint):
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
            model._fqn2spec_info[name] = SpecInfo("ep", Shard(0), name, mesh, fsdp_shard_dim=shard_dim)

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
@pytest.mark.parametrize("shard_dim", [0, 1])
def test_lora_adapter_export_preserves_expert_shard_layout(world, shard_dim, tmp_path):
    spawn_with_rendezvous(_lora_export_worker, args=(world, shard_dim, str(tmp_path / "adapter")), nprocs=world)
