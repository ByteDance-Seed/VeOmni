# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Native NCCL/HCCL regression for consistently ordered metric reductions."""

import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.callbacks.trace_callback import EnvironMeterCallback
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


def _accelerator_plan(context):
    if context.capabilities.backend != "cpu" and len(context.capabilities.devices) >= 2:
        return LaunchSpec(device="accelerator", nproc=2)
    return LaunchSpec(device="cpu")


def _metric_schema_accelerator_worker(rank, expected_device, endpoint):
    device = get_device_type()
    assert device == expected_device
    expected_backend = get_dist_comm_backend()
    assert expected_backend != "gloo"
    if expected_device == "npu":
        assert expected_backend == "hccl"
    get_torch_device().set_device(rank)
    dist.init_process_group(
        get_dist_comm_backend(), store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        assert dist.get_backend() == expected_backend
        callback = object.__new__(EnvironMeterCallback)
        callback.parallel_state = SimpleNamespace(fsdp_group=None)
        callback.start_time = time.time()
        callback.lora_config = None
        callback.freeze_vit = None
        callback.trainer = SimpleNamespace(
            environ_meter=SimpleNamespace(step=lambda *args, **kwargs: {}),
            model=SimpleNamespace(lr_scheduler=None),
        )
        metrics = {"x": 1.0, "y": 10.0} if rank == 0 else {"y": 2.0, "x": 20.0}
        callback.on_step_end(TrainerState(global_step=1), loss=0.0, loss_dict={}, grad_norm=0.0, aux_metrics=metrics)
        assert callback.trainer.step_train_metrics["training/x"] == pytest.approx(10.5)
        assert callback.trainer.step_train_metrics["training/y"] == pytest.approx(6.0)
        alive = torch.ones(1, device=get_device_type())
        dist.all_reduce(alive)
        assert alive.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_accelerator_plan)
def test_metric_reduction_accelerator():
    assigned_backend = runner_capabilities().backend if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    require_npu = os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1"
    if require_npu and assigned_backend is not None and assigned_backend != "npu":
        pytest.fail("NPU test requested with a non-NPU CI assignment")
    required = (assigned_backend is not None and assigned_backend != "cpu") or require_npu
    device = get_device_type()
    expected_device = assigned_backend or ("npu" if require_npu else device)
    available = (
        expected_device != "cpu"
        and device == expected_device
        and get_torch_device().is_available()
        and get_torch_device().device_count() >= 2
    )
    if not available:
        if required:
            pytest.fail("Assigned or required accelerator runtime must provide two usable devices")
        pytest.skip("requires two accelerator devices")
    spawn_with_rendezvous(_metric_schema_accelerator_worker, args=(expected_device,), nprocs=2)
