# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Expert rank-local FP32 optimizer offload survives a fresh-process restore."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.distributed.parallel_plan import SpecInfo
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
from veomni.optim.optimizer import MultiOptimizer


def _worker(rank, root, phase, persistent):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init-{phase}", rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep_fsdp", "ep"))
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ep",), extra_parallel_fsdp_device_mesh={"ep": mesh}
        )
        model = torch.nn.Module()
        model.experts = torch.nn.Module()
        initial = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8) / 16
        if persistent:
            weight = distribute_tensor(initial, mesh, [Shard(1), Shard(0)])
            spec = SpecInfo("ep", Shard(0), "experts.weight", mesh, 1, 1)
        else:
            ep_slice = initial.chunk(2, dim=0)[mesh.get_local_rank("ep")].contiguous()
            weight = distribute_tensor(ep_slice, mesh["ep_fsdp"], [Shard(1)])
            spec = SpecInfo("ep", Shard(0), "experts.weight", mesh, 1)
        model.experts.weight = torch.nn.Parameter(weight)
        model._fqn2spec_info = {"experts.weight": spec}
        leaf = CPUOffloadAdamW([{"params": [model.experts.weight], "cpu_offload": True}], lr=0.01)
        assert id(model.experts.weight) in leaf._offloaded
        optimizer = MultiOptimizer(model, {"experts": leaf}, ["experts"])
        payload = {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}

        def snapshot():
            state = leaf.state[model.experts.weight]
            captured = {}
            for key in ("master_weight", "exp_avg", "exp_avg_sq", "step"):
                value = state[key].to_local() if hasattr(state[key], "to_local") else state[key]
                assert value.device.type == "cpu" and value.dtype == torch.float32
                captured[key] = value.detach().clone()
            captured["weight"] = model.experts.weight.to_local().detach().clone()
            return captured

        if phase == "save":
            model.experts.weight.grad = torch.full_like(model.experts.weight, 0.1 + rank * 0.05)
            optimizer.step()
            optimizer.zero_grad()
            DistributedCheckpointer.save(f"{root}/checkpoint", payload, parallel_state=parallel)
            before = snapshot()
            model.experts.weight.grad = torch.full_like(model.experts.weight, 0.2 + rank * 0.05)
            optimizer.step()
            torch.save({"before": before, "after": snapshot()}, f"{root}/reference-{rank}.pt")
        else:
            DistributedCheckpointer.load(f"{root}/checkpoint", payload, parallel_state=parallel)
            reference = torch.load(f"{root}/reference-{rank}.pt", weights_only=True)
            torch.testing.assert_close(snapshot(), reference["before"], rtol=0, atol=0)
            model.experts.weight.grad = torch.full_like(model.experts.weight, 0.2 + rank * 0.05)
            optimizer.step()
            torch.testing.assert_close(snapshot(), reference["after"], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
@pytest.mark.parametrize("persistent", [False, True])
def test_expert_fp32_offload_cold_resume_and_update(tmp_path: Path, persistent):
    mp.spawn(_worker, args=(str(tmp_path), "save", persistent), nprocs=4, join=True)
    mp.spawn(_worker, args=(str(tmp_path), "load", persistent), nprocs=4, join=True)
