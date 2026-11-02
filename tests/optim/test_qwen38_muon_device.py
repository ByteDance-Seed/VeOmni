# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Device update and exact-state recovery for mixed Qwen Muon/AdamW."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer
from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans
from veomni.utils.device import get_device_type, get_torch_device


def _require_native_device():
    capabilities = runner_capabilities()
    if capabilities.backend == "cpu":
        pytest.skip("Requires a native CUDA or Ascend device")
    runtime = get_torch_device()
    if capabilities.backend != get_device_type() or not runtime.is_available() or runtime.device_count() < 1:
        pytest.fail("Assigned accelerator device is unavailable")
    return torch.device(capabilities.backend)


def _build(device, dtype):
    model = torch.nn.Module()
    model.down_proj = torch.nn.Linear(4, 4, bias=False, device=device, dtype=dtype)
    model.lm_head = torch.nn.Linear(4, 4, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            values = (torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) + index) / 20
            param.copy_(values.to(device=device, dtype=dtype))
    plans = add_qwen38_plans(model, [{"params": list(model.parameters()), "cpu_offload": True}])
    leaf = Qwen38MuonAdamW(plans, lr=1e-3, betas=(0.9, 0.95), weight_decay=0)
    return model, leaf, MultiOptimizer(model, {"mixed": leaf}, ["mixed"])


def _give_gradients(model):
    for index, param in enumerate(model.parameters()):
        values = (torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) - 5 + index) / 16
        param.grad = values.to(device=param.device, dtype=param.dtype)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_qwen_muon_adam_offload_update_and_recovery(tmp_path, dtype):
    device = _require_native_device()
    if device.type == "cuda":
        from veomni.optim.qwen38_muon import _check_cuda_fp32_policy

        _check_cuda_fp32_policy()
    model, leaf, optimizer = _build(device, dtype)
    reference, reference_leaf, reference_optimizer = _build(torch.device("cpu"), dtype)
    for _ in range(2):
        _give_gradients(model)
        _give_gradients(reference)
        optimizer.step()
        reference_optimizer.step()
    optimizer.zero_grad()
    reference_optimizer.zero_grad()
    for param, ref_param in zip(model.parameters(), reference.parameters(), strict=True):
        for key in ("master_weight", "exp_avg", "exp_avg_sq"):
            actual, expected = leaf.state[param][key], reference_leaf.state[ref_param][key]
            assert actual.device.type == "cpu" and actual.dtype == torch.float32
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    target, target_leaf, restored = _build(device, dtype)
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": target, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    for src, dst in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(src, dst, rtol=0, atol=0)
        for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(leaf.state[src][key], target_leaf.state[dst][key], rtol=0, atol=0)
    _give_gradients(model)
    _give_gradients(target)
    optimizer.step()
    restored.step()
    for src, dst in zip(model.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(src, dst, rtol=1e-5, atol=1e-6)
        for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(leaf.state[src][key], target_leaf.state[dst][key], rtol=1e-5, atol=1e-6)
