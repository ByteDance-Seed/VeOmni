# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""TP/FSDP checkpoint provenance must not weaken unknown expert guards."""

import os
from copy import copy, deepcopy
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._state_dict_utils import _create_cpu_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.distributed.tensor.placement_types import _StridedShard

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.checkpoint.dcp_checkpointer import (
    _EXPERT_LAYOUTS_KEY,
    DistributedCheckpointer,
    _ExtraParallelSavePlanner,
    _ModelStrictLoadPlanner,
    _tp_only_source_layout,
    _tp_only_source_parameters,
    dcp_to_torch_state_dict,
)
from veomni.checkpoint.layout import optimizer_dir, weights_dir
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state
from veomni.utils.device import IS_NPU_AVAILABLE


def _tp_fsdp_worker(rank, world, directory, device_type, endpoint):
    device = torch.device(device_type, rank) if device_type == "npu" else torch.device("cpu")
    if device_type == "npu":
        torch.npu.set_device(device)
    dist.init_process_group(
        "hccl" if device_type == "npu" else "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    try:
        ambient = _init_parallel_state(dp_size=4, device_type=device_type, name="base")
        state = _init_parallel_state(dp_size=2, tp_size=2, device_type=device_type, name="policy")
        assert get_parallel_state() is ambient and state is not ambient
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
        reference = deepcopy(model)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
        model.to(device)
        parallelize_module(model, state.tp_mesh, {"0": ColwiseParallel()})
        fully_shard(model, mesh=state.fsdp_mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        inputs = torch.arange(16, dtype=torch.float32).view(2, 8) / 16
        model(inputs.to(device)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference(inputs).sum().backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        param = model[0].weight
        assert isinstance(param, DTensor)
        assert param.placements == (_StridedShard(0, split_factor=2), Shard(0))
        owners, mesh = _tp_only_source_parameters(model, state)
        assert owners["0.weight"] is param
        layout = _tp_only_source_layout("model.0.weight", param, owners, mesh)
        assert layout and layout["expert_order"] == "canonical"

        # Optimizer-shaped state must map to the same parameter and layout.
        avg = optimizer.state[param]["exp_avg"]
        variance = optimizer.state[param]["exp_avg_sq"]
        torch.testing.assert_close(param.full_tensor().cpu(), reference[0].weight, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            avg.full_tensor().cpu(), reference_optimizer.state[reference[0].weight]["exp_avg"], atol=1e-6, rtol=1e-5
        )
        assert _tp_only_source_layout("optimizer.state.0.weight.exp_avg", avg, owners, mesh) == layout
        assert _tp_only_source_layout("optimizer.state.missing.exp_avg", avg, owners, mesh) is None
        wrong_placement = DTensor.from_local(
            torch.zeros_like(param.to_local()), mesh, [Partial(), Shard(0)], shape=param.shape, stride=param.stride()
        )
        assert _tp_only_source_layout("model.0.weight", wrong_placement, owners, mesh) is None
        wrong_split = DTensor.from_local(
            torch.zeros_like(param.to_local()),
            mesh,
            [_StridedShard(0, split_factor=3), Shard(0)],
            shape=param.shape,
            stride=param.stride(),
        )
        assert _tp_only_source_layout("model.0.weight", wrong_split, owners, mesh) is None
        other_root = DeviceMesh(device_type, [[0, 1], [2, 3]], mesh_dim_names=mesh.mesh_dim_names, _init_backend=False)
        wrong_root = DTensor.from_local(
            torch.zeros_like(param.to_local()), other_root, param.placements, shape=param.shape, stride=param.stride()
        )
        assert _tp_only_source_layout("model.0.weight", wrong_root, owners, mesh) is None
        assert _tp_only_source_layout("model.0.weight", wrong_root, owners, mesh, allow_staged_copies=True) is None
        staged = _create_cpu_state_dict({"weight": param})["weight"]
        if staged.device_mesh._get_root_mesh() is not mesh._get_root_mesh():
            assert _tp_only_source_layout("model.0.weight", staged, owners, mesh) is None
        assert _tp_only_source_layout("model.0.weight", staged, owners, mesh, allow_staged_copies=True) == layout
        assert _tp_only_source_parameters(None, state) == ({}, None)
        assert _tp_only_source_parameters(model, copy(state)) == ({}, None)
        with patch.object(dist, "is_initialized", return_value=False):
            assert _tp_only_source_parameters(model, state) == ({}, None)
        extra_state = _init_parallel_state(
            dp_size=2, tp_size=2, extra_parallel_sizes=(2,), device_type=device_type, name="policy_ep"
        )
        assert _tp_only_source_parameters(model, extra_state) == ({}, None)
        aliased = torch.nn.Module()
        aliased.left = model[0]
        aliased.right = model[0]
        assert not _tp_only_source_parameters(aliased, state)[0]

        checkpoint = directory + "/dense"
        dcp.save(
            {
                "model": {"0.weight": param},
                "optimizer": {"state": {"0.weight": {"exp_avg": avg, "exp_avg_sq": variance}}},
            },
            checkpoint_id=checkpoint,
            planner=_ExtraParallelSavePlanner(source_model=model, parallel_state=state),
        )
        metadata = dcp.FileSystemReader(checkpoint).read_metadata().planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
        assert metadata["model.0.weight"]["expert_order"] == "canonical"
        assert metadata["optimizer.state.0.weight.exp_avg"]["expert_order"] == "canonical"
        expected = param.full_tensor().cpu()
        target = {"model": {"0.weight": torch.zeros_like(expected)}}
        dcp.load(target, checkpoint_id=checkpoint, planner=_ModelStrictLoadPlanner(strict_model=True))
        torch.testing.assert_close(target["model"]["0.weight"], expected, atol=0, rtol=0)
        if rank == 0:
            exported = dcp_to_torch_state_dict(checkpoint)
            torch.testing.assert_close(exported["0.weight"], expected, atol=0, rtol=0)
        dist.barrier()

        production = directory + "/production"
        DistributedCheckpointer.save(production, {"model": model, "optimizer": optimizer}, parallel_state=state)
        for path, key in (
            (weights_dir(production), "model.0.weight"),
            (optimizer_dir(production), "optimizer.state.0.weight.exp_avg"),
            (optimizer_dir(production), "optimizer.state.0.weight.exp_avg_sq"),
        ):
            recorded = dcp.FileSystemReader(path).read_metadata().planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
            assert recorded[key]["expert_order"] == "canonical"

        torch.manual_seed(9)
        restored_model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
        restored_model.to(device)
        parallelize_module(restored_model, state.tp_mesh, {"0": ColwiseParallel()})
        fully_shard(restored_model, mesh=state.fsdp_mesh)
        restored_param = restored_model[0].weight
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
        DistributedCheckpointer.load(
            production, {"model": restored_model, "optimizer": restored_optimizer}, parallel_state=state
        )
        restored_avg = restored_optimizer.state[restored_param]["exp_avg"]
        restored_variance = restored_optimizer.state[restored_param]["exp_avg_sq"]
        torch.testing.assert_close(restored_param.to_local(), param.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_avg.to_local(), avg.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_variance.to_local(), variance.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_optimizer.state[restored_param]["step"], optimizer.state[param]["step"])

        async_path = directory + "/production_async"
        DistributedCheckpointer.save(
            async_path, {"model": model, "optimizer": optimizer}, parallel_state=state, save_async=True
        )
        DistributedCheckpointer.wait_for_pending_save()
        for path, key in (
            (weights_dir(async_path), "model.0.weight"),
            (optimizer_dir(async_path), "optimizer.state.0.weight.exp_avg"),
            (optimizer_dir(async_path), "optimizer.state.0.weight.exp_avg_sq"),
        ):
            recorded = dcp.FileSystemReader(path).read_metadata().planner_data[_EXPERT_LAYOUTS_KEY]["tensors"]
            assert recorded[key]["expert_order"] == "canonical"
        DistributedCheckpointer.load(
            async_path, {"model": restored_model, "optimizer": restored_optimizer}, parallel_state=state
        )
        restored_avg = restored_optimizer.state[restored_param]["exp_avg"]
        restored_variance = restored_optimizer.state[restored_param]["exp_avg_sq"]
        torch.testing.assert_close(restored_param.to_local(), param.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_avg.to_local(), avg.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_variance.to_local(), variance.to_local(), atol=0, rtol=0)
        model(inputs.to(device)).sum().backward()
        restored_model(inputs.to(device)).sum().backward()
        reference(inputs).sum().backward()
        optimizer.step()
        restored_optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(param.full_tensor().cpu(), reference[0].weight, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(restored_param.to_local(), param.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_avg.to_local(), avg.to_local(), atol=0, rtol=0)
        torch.testing.assert_close(restored_variance.to_local(), variance.to_local(), atol=0, rtol=0)

        # Even an innocently named tensor must fail closed without a proven owner.
        unknown = directory + "/unknown"
        dcp.save(
            {"model": {"unknown": param}},
            checkpoint_id=unknown,
            planner=_ExtraParallelSavePlanner(source_model=model, parallel_state=state),
        )
        target = {"model": {"unknown": torch.zeros_like(expected)}}
        with pytest.raises(dcp.CheckpointException, match="noncanonical expert order"):
            dcp.load(target, checkpoint_id=unknown, planner=_ModelStrictLoadPlanner(strict_model=True))
        if rank == 0:
            with pytest.raises(RuntimeError, match="noncanonical expert order"):
                dcp_to_torch_state_dict(unknown)
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_tp_fsdp_checkpoint_requires_provenance(tmp_path):
    spawn_with_rendezvous(_tp_fsdp_worker, args=(4, str(tmp_path), "cpu"), nprocs=4)


def _npu_launch_plan(context):
    if context.capabilities.backend != "npu":
        # Non-NPU runners need no accelerator lease to skip (or to fail
        # explicitly when VEOMNI_REQUIRE_NPU_TESTS=1).
        return LaunchSpec(device="cpu")
    return LaunchSpec(nproc=4)


@pytest.mark.launch_plan(provider=_npu_launch_plan)
def test_tp_fsdp_checkpoint_native_npu(tmp_path):
    ci_backend = runner_capabilities().backend if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    require_npu = os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1" or ci_backend == "npu"
    available = ci_backend in (None, "npu") and IS_NPU_AVAILABLE and torch.npu.is_available()
    if not available:
        if require_npu:
            pytest.fail("NPU CI or an explicit NPU test requires an available NPU")
        pytest.skip("native TP/FSDP requires NPU")
    if torch.npu.device_count() < 4:
        pytest.fail("native TP/FSDP requires four visible NPUs")
    spawn_with_rendezvous(_tp_fsdp_worker, args=(4, str(tmp_path), "npu"), nprocs=4)
