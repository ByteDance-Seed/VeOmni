"""Muon communication checks respect FSDP groups and independent EP owners."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim.qwen38_muon import _validate_file_muon_gradient_traffic


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _worker(rank, world_size, rendezvous, mode):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=20)
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep", "fsdp"))
        full = torch.ones((4, 4), dtype=torch.float32)
        tensor = distribute_tensor(full, mesh, [Replicate(), Shard(0)])
        param = torch.nn.Parameter(tensor)
        grad = distribute_tensor(full, mesh, [Replicate(), Shard(0)])
        if mode.startswith("group-identity"):
            alternate = {}
            for members in ((0, 1), (2, 3)):
                group = dist.new_group(list(members))
                if rank in members:
                    alternate[members] = group
            original = mesh.get_group
            if mode == "group-identity-all" or rank % 2:

                def get_group(axis):
                    old = original(axis)
                    ranks = tuple(dist.get_process_group_ranks(old))
                    return alternate.get(ranks, old)

                mesh.get_group = get_group
        if mode == "same-group-mismatch":
            present = rank != 1
        elif mode == "different-owner":
            present = rank < 2
        else:
            present = True
        param.grad = grad if present else None
        groups = [{"params": [param], "qwen38_plan": json.dumps([["muon", 0, 4]])}]
        failed = False
        try:
            _validate_file_muon_gradient_traffic(groups)
        except RuntimeError:
            failed = True
        result = torch.tensor(int(failed))
        dist.all_reduce(result)
        assert int(result) == (world_size if mode in ("same-group-mismatch", "group-identity-mismatch") else 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize(
    "mode", ["all", "same-group-mismatch", "different-owner", "group-identity-all", "group-identity-mismatch"]
)
def test_muon_grad_participation_uses_actual_fsdp_group(tmp_path, mode):
    mp.start_processes(_worker, args=(4, str(tmp_path / "rendezvous"), mode), nprocs=4, join=True, start_method="fork")


def _state_failure_worker(rank, world_size, rendezvous, root, mode):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from veomni.optim import optimizer as factory
    from veomni.optim.ple_world_profile import establish
    from veomni.utils import dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=15)
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        model = torch.nn.Module()
        model.attn = torch.nn.Module()
        model.attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        param = torch.nn.Parameter(distribute_tensor(torch.ones(4, 4), mesh, [Shard(0)]))
        model.attn.o_proj.weight = param
        param.grad = distribute_tensor(torch.ones(4, 4), mesh, [Shard(0)])
        config = SimpleNamespace(
            type="qwen38_muon",
            qwen38_ple_moments_world_profile=True,
            qwen38_ple_moments_scratch_dir=root,
            cpu_offload_param_patterns=[".+"],
            cpu_offload_cpu_moment_dtype="float32",
            cpu_offload_resident_moment_dtype="float32",
        )
        receipt = establish(config, SimpleNamespace(), SimpleNamespace(save_async=False))
        factory._should_build_extra_parallel_aware = lambda model: False
        optimizer = factory.build_optimizer(
            model,
            lr=0.001,
            weight_decay=0.0,
            optimizer_type="qwen38_muon",
            optimizer_config=config,
            ple_profile_receipt=receipt,
            param_groups=[{"params": [param], "weight_decay": 0.0}],
        )
        if mode != "lazy":
            optimizer.state[param] = optimizer._initialize_state(param, torch.device("cpu"))
            if rank == 0 and mode == "state-dtype":
                optimizer.state[param]["exp_avg"] = optimizer.state[param]["exp_avg"].bfloat16()
            if rank == 0 and mode == "missing-state":
                del optimizer.state[param]["exp_avg"]
            if rank == 0 and mode == "step-noninteger":
                optimizer.state[param]["step"].fill_(0.5)
            if rank == 0 and mode == "step-mismatch":
                optimizer.state[param]["step"].fill_(2.0)
        failed = False
        try:
            optimizer.step()
        except RuntimeError:
            failed = True
        result = torch.tensor(int(failed))
        dist.all_reduce(result)
        assert int(result) == (0 if mode == "lazy" else world_size)
        if mode != "lazy":
            assert optimizer._ple_state_poisoned
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("mode", ["lazy", "state-dtype", "missing-state", "step-noninteger", "step-mismatch"])
def test_file_muon_state_errors_fail_before_peer_collectives(tmp_path, mode):
    mp.start_processes(
        _state_failure_worker,
        args=(2, str(tmp_path / "state-rendezvous"), str(tmp_path), mode),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_cpu_muon_remains_usable_when_accelerator_is_visible(monkeypatch):
    from veomni.optim import qwen38_muon

    parameter = torch.nn.Parameter(torch.ones(2, 2))
    group = {
        "params": [parameter],
        "qwen38_plan": json.dumps([["muon", 0, 2]]),
        "qwen38_recipe": qwen38_muon.RECIPE,
        "weight_decay": 0.0,
    }
    from veomni.utils import device

    monkeypatch.setattr(device, "get_device_type", lambda: "npu")
    qwen38_muon.Qwen38MuonAdamW([group])
