# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""New EP1 checkpoints preserve the source layout across a fresh-process resume."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import (
    _EXPERT_LAYOUTS_KEY,
    DistributedCheckpointer,
    _ExtraParallelSavePlanner,
    _ModelStrictLoadPlanner,
    _no_ep_expert_layout,
    dcp_to_torch_state_dict,
)
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.optim.optimizer import MultiOptimizer


def _build_noep_model(parallel_state, optimizer_kind, real_fsdp=False):
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.experts = torch.nn.Module()
    mesh = parallel_state.fsdp_mesh
    placements = [Replicate()] * (mesh.ndim - 1) + [Shard(0)]
    for name, shape in (("bias", (8,)), ("matrix", (8, 4)), ("stack", (8, 4, 2))):
        value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 100
        if not real_fsdp:
            value = distribute_tensor(value, mesh, placements)
        setattr(model.layer.experts, name, torch.nn.Parameter(value))
    if real_fsdp:
        from torch.distributed.fsdp import fully_shard

        fully_shard(model, mesh=mesh)
    leaf = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95), foreach=False)
    optimizer = MultiOptimizer(model, {"expert": leaf}, ["expert"]) if optimizer_kind == "multi" else leaf
    return model, optimizer, leaf


def _local(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu().clone() if torch.is_tensor(tensor) else tensor


def _snapshot(model, leaf):
    result = {}
    for name, parameter in model.named_parameters():
        result[name] = {
            "weight": _local(parameter),
            "state": {key: _local(value) for key, value in leaf.state[parameter].items()},
        }
    return result


def _step(model, optimizer, value):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _noep_worker(rank, directory, geometry, optimizer_kind, phase, endpoint):
    world = 4 if geometry == "hsdp" else 2
    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world, timeout=timedelta(seconds=60)
    )
    try:
        geometry_args = {
            "fsdp": dict(dp_size=2, dp_shard_size=2),
            "fsdp2": dict(dp_size=2, dp_shard_size=2),
            "sp": dict(dp_size=1, dp_shard_size=1, ulysses_size=2),
            "hsdp": dict(dp_size=4, dp_replicate_size=2, dp_shard_size=2),
        }[geometry]
        parallel_state = _init_parallel_state(device_type="cpu", **geometry_args)
        model, optimizer, leaf = _build_noep_model(parallel_state, optimizer_kind, real_fsdp=geometry == "fsdp2")
        checkpoint = str(Path(directory) / "checkpoint")
        state = {"model": model, "optimizer": optimizer}
        if optimizer_kind == "multi":
            state["_veomni_scheduler_object"] = None
        if phase == "save":
            _step(model, optimizer, 0.25)
            DistributedCheckpointer.save(checkpoint, state, parallel_state=parallel_state)
            reference_before = _snapshot(model, leaf)
            full_model = {name: parameter.detach().full_tensor().cpu() for name, parameter in model.named_parameters()}
            _step(model, optimizer, -0.125)
            torch.save(
                {"before": reference_before, "after": _snapshot(model, leaf)},
                Path(directory) / f"reference-{rank}.pt",
            )
            if rank == 0:
                torch.save(full_model, Path(directory) / "full-model.pt")
        else:
            reference = torch.load(Path(directory) / f"reference-{rank}.pt", weights_only=True)
            DistributedCheckpointer.load(checkpoint, state, parallel_state=parallel_state)
            torch.testing.assert_close(_snapshot(model, leaf), reference["before"], rtol=0, atol=0)
            _step(model, optimizer, -0.125)
            torch.testing.assert_close(_snapshot(model, leaf), reference["after"], rtol=0, atol=0)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("geometry", ["fsdp", "fsdp2", "sp", "hsdp"])
@pytest.mark.parametrize("optimizer_kind", ["adamw", "multi"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_new_noep_expert_checkpoint_fresh_process_roundtrip(tmp_path, geometry, optimizer_kind):
    world = 4 if geometry == "hsdp" else 2
    spawn_with_rendezvous(_noep_worker, args=(str(tmp_path), geometry, optimizer_kind, "save"), nprocs=world)
    model_dir = tmp_path / "checkpoint" / "model" / "ckpt"
    optimizer_dir = tmp_path / "checkpoint" / "model" / "optimizer"
    for root, prefix in ((model_dir, "model.layer.experts."), (optimizer_dir, "optimizer.state.layer.experts.")):
        layouts = dcp.FileSystemReader(root).read_metadata().planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
        for name in ("bias", "matrix", "stack"):
            key = prefix + name + (".exp_avg" if root == optimizer_dir else "")
            assert layouts[key]["source_kind"] == "veomni_tp1_no_ep"
            assert layouts[key]["expert_order"] == "canonical"
    exported = dcp_to_torch_state_dict(model_dir)
    expected = torch.load(tmp_path / "full-model.pt", weights_only=True)
    torch.testing.assert_close(exported, expected, rtol=0, atol=0)
    spawn_with_rendezvous(_noep_worker, args=(str(tmp_path), geometry, optimizer_kind, "resume"), nprocs=world)


def _unproven_source_worker(rank, directory, mode, endpoint):
    world = 4 if mode == "tp" else 2
    dist.init_process_group(
        "gloo", store=endpoint.connect(), rank=rank, world_size=world, timeout=timedelta(seconds=60)
    )
    try:
        if mode == "ep":
            parallel_state = _init_parallel_state(
                dp_size=2, dp_shard_size=2, extra_parallel_sizes=(2,), device_type="cpu"
            )
        elif mode == "tp":
            parallel_state = _init_parallel_state(dp_size=2, dp_shard_size=2, tp_size=2, device_type="cpu")
        else:
            parallel_state = _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu")
        mesh = parallel_state.fsdp_mesh
        if mode == "custom":
            mesh = DeviceMesh("cpu", list(range(world)), mesh_dim_names=("custom_axis",))
        value = distribute_tensor(torch.arange(32, dtype=torch.float32).reshape(8, 4), mesh, [Shard(0)])
        key = "model.layer.experts.weight"
        assert _no_ep_expert_layout(key, value, parallel_state) is None
        path = str(Path(directory) / "unproven")
        dcp.save(
            {"model": {"layer.experts.weight": value}},
            checkpoint_id=path,
            planner=_ExtraParallelSavePlanner(parallel_state=parallel_state),
        )
        layouts = dcp.FileSystemReader(path).read_metadata().planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
        assert layouts[key]["expert_order"] == "unproven"
        target = {"model": {"layer.experts.weight": torch.zeros(8, 4)}}
        with pytest.raises(dcp.CheckpointException, match="noncanonical expert order"):
            dcp.load(target, checkpoint_id=path, planner=_ModelStrictLoadPlanner(strict_model=True))
        assert target["model"]["layer.experts.weight"].count_nonzero() == 0
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ep", "tp", "custom"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=4))
def test_ep_tp_and_custom_mesh_are_not_inferred_as_noep(tmp_path, mode):
    spawn_with_rendezvous(_unproven_source_worker, args=(str(tmp_path), mode), nprocs=4 if mode == "tp" else 2)


def _uninitialized_parallel_state_worker(rank, directory, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=1, timeout=timedelta(seconds=30))
    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))

        def build():
            model = torch.nn.Module()
            model.layer = torch.nn.Module()
            model.layer.experts = torch.nn.Module()
            model.layer.experts.bias = torch.nn.Parameter(
                distribute_tensor(torch.arange(8, dtype=torch.float32), mesh, [Shard(0)])
            )
            return model

        model = build()
        checkpoint = str(Path(directory) / "uninitialized")
        DistributedCheckpointer.save(checkpoint, {"model": model})
        target = build()
        DistributedCheckpointer.load(checkpoint, {"model": target})
        torch.testing.assert_close(
            target.layer.experts.bias.to_local(), model.layer.experts.bias.to_local(), rtol=0, atol=0
        )
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=1))
def test_uninitialized_parallel_state_preserves_single_rank_save(tmp_path):
    spawn_with_rendezvous(_uninitialized_parallel_state_worker, args=(str(tmp_path),), nprocs=1)
