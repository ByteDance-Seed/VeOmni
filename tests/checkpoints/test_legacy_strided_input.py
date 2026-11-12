# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Legacy FSDP-first DTensor input remains supported by the drop helper."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import drop_extra_parallel_dim_v2


def _worker(rank, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        for geometry, rows in (((2, 2), 6), ((4, 1), 5)):
            mesh = init_device_mesh("cpu", geometry, mesh_dim_names=("ep_fsdp", "ep"))
            fsdp = mesh["ep_fsdp"]
            full = torch.arange(rows * 7, dtype=torch.float32).reshape(rows, 7)
            ep_slice = full.chunk(geometry[1], dim=0)[mesh.get_local_rank("ep")]
            local = distribute_tensor(ep_slice.contiguous(), fsdp, [Shard(0)])
            first = _StridedShard(0, split_factor=2) if geometry[1] == 2 else Shard(0)
            saved = DTensor.from_local(
                local.to_local(), mesh, [first, Shard(0)], run_check=False, shape=full.shape, stride=full.stride()
            )
            restored = drop_extra_parallel_dim_v2(saved, fsdp, 0, extra_parallel_name="ep")
            assert tuple(restored.shape) == tuple(local.shape)
            torch.testing.assert_close(restored.to_local(), local.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_legacy_fsdp_first_strided_and_ep1_drop():
    spawn_with_rendezvous(_worker, args=(), nprocs=4)
