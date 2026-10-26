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
from veomni.checkpoint.dcp_checkpointer import (
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
            dcp.save({"model": {"weight": source}}, checkpoint_id=path, planner=_ExtraParallelSavePlanner())
            target = {"model": {"weight": wrap(1 - source_dim, empty=True)}}
            dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
            loaded = target["model"]["weight"]
            torch.testing.assert_close(loaded.full_tensor(), global_weight, rtol=0, atol=0)
            local_dt = drop_extra_parallel_dim(loaded, submesh, 1 - source_dim)
            torch.testing.assert_close(local_dt.full_tensor(), ep_weight, rtol=0, atol=0)

        # Legacy metadata used two ordinary Shards, recording the wrong order.
        local = ep_weight.chunk(2, 0)[fsdp_rank].contiguous()
        placements = [Shard(0), Shard(0)] if world == 4 else [Replicate(), Shard(0), Shard(0)]
        legacy = DTensor.from_local(local, mesh, placements)
        for key, strict in [("model", True), ("model", False), ("optimizer", False)]:
            path = directory + key + str(strict)
            dcp.save({key: {"weight": legacy}}, checkpoint_id=path)
            target = {key: {"weight": wrap(0, empty=True)}}
            with pytest.raises(dcp.CheckpointException, match="unversioned checkpoint"):
                dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=strict))
            assert target[key]["weight"].to_local().count_nonzero() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=from_parameter("world", device="cpu"))
@pytest.mark.parametrize("world", [4, 8])
def test_expert_shard_layout_roundtrip_and_legacy_rejection(world, tmp_path):
    mp.spawn(
        _layout_worker, args=(world, str(tmp_path / "dcp"), f"file://{tmp_path / 'rdzv'}"), nprocs=world, join=True
    )
