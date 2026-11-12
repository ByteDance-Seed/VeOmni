# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

from datetime import timedelta
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


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _worker(rank, root, phase):
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/ple-{phase}-init", rank=rank, world_size=4, timeout=timedelta(seconds=40)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ple_fsdp", "ple"))
        owner = mesh.get_local_rank("ple")
        parallel = SimpleNamespace(
            dp_mode="fsdp2", extra_parallel_names=("ple",), extra_parallel_fsdp_device_mesh={"ple": mesh}
        )
        model = torch.nn.Module()
        initial = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
        model.register_parameter("weight", torch.nn.Parameter(distribute_tensor(initial, mesh, [Shard(1), Shard(0)])))
        model._fqn2spec_info = {"weight": SpecInfo("ple", Shard(0), "weight", mesh, 1, 1)}
        leaf = CPUOffloadAdamW([{"params": [model.weight], "cpu_offload": True}], lr=0.01)
        optimizer = MultiOptimizer(model, {"ple": leaf}, ["ple"])

        def snapshot():
            state = leaf.state.get(model.weight, {})
            return {
                "weight": model.weight.to_local().detach().clone(),
                "state": {
                    key: (value.to_local() if hasattr(value, "to_local") else value).detach().clone()
                    for key, value in state.items()
                },
            }

        if phase == "save":
            if owner == 0:
                model.weight.grad = torch.full_like(model.weight, 0.1)
                optimizer.step()
                optimizer.zero_grad()
            DistributedCheckpointer.save(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
            before = snapshot()
            model.weight.grad = torch.full_like(model.weight, 0.2)
            optimizer.step()
            torch.save({"before": before, "after": snapshot()}, f"{root}/ple-ref-{rank}.pt")
        else:
            DistributedCheckpointer.load(
                f"{root}/checkpoint",
                {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
                parallel_state=parallel,
            )
            reference = torch.load(f"{root}/ple-ref-{rank}.pt", weights_only=True)
            torch.testing.assert_close(snapshot(), reference["before"], rtol=0, atol=0)
            model.weight.grad = torch.full_like(model.weight, 0.2)
            optimizer.step()
            torch.testing.assert_close(snapshot(), reference["after"], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_persistent_ple_fp32_cpu_optimizer_fresh_process(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path), "save"), nprocs=4, join=True)
    mp.spawn(_worker, args=(str(tmp_path), "load"), nprocs=4, join=True)
