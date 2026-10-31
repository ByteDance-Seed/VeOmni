# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Native staging coverage, separate from the CPU-only optimizer suite."""

import contextlib
import os
import weakref
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.cpu_offload_adamw import CPUOffloadAdamW
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


def _require_native_device():
    """Only optional CPU runs may skip a native test."""
    capabilities = runner_capabilities() if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        if capabilities is not None and capabilities.backend != "cpu":
            pytest.fail("Assigned accelerator runtime is unavailable")
        pytest.skip("Requires a native CUDA or Ascend device")
    device_type = get_device_type()
    if capabilities is not None and capabilities.backend != "cpu" and device_type != capabilities.backend:
        pytest.fail("Assigned accelerator backend does not match the runtime")
    runtime = get_torch_device()
    if not runtime.is_available() or runtime.device_count() < 1:
        pytest.fail("Assigned accelerator has no usable device")
    return torch.device(device_type)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
def test_accelerator_offload_staging_matches_fp32_reference():
    """Exercise real chunked D2H/update/H2D transfers on one CUDA or Ascend device."""
    device = _require_native_device()
    generator = torch.Generator(device="cpu").manual_seed(3)
    initial = torch.randn(257, dtype=torch.bfloat16, generator=generator, device="cpu")
    param = torch.nn.Parameter(initial.to(device))
    reference = torch.nn.Parameter(initial.float())
    # Multiple full chunks and a partial tail must reuse the staging buffer.
    optimizer = CPUOffloadAdamW([{"params": [param], "cpu_offload": True}], lr=1e-3, chunk_size=64)
    expected = torch.optim.AdamW([reference], lr=1e-3, betas=(0.9, 0.95), weight_decay=0, foreach=False)
    for _ in range(3):
        gradient = torch.randn(initial.shape, dtype=initial.dtype, generator=generator, device="cpu")
        param.grad = gradient.to(device)
        reference.grad = gradient.float()
        optimizer.step()
        expected.step()
        assert param.device.type == device.type
        assert param.grad.device == param.device
        state = optimizer.state[param]
        for key in ("master_weight", "exp_avg", "exp_avg_sq"):
            assert state[key].device.type == "cpu"
            assert state[key].dtype == torch.float32
        torch.testing.assert_close(state["master_weight"], reference, rtol=1e-6, atol=1e-7)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[key], expected.state[reference][key], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(param.detach().cpu(), reference.bfloat16(), rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nondefault_stream", [False, True])
def test_accelerator_offload_reuses_and_grows_staging_across_parameters(dtype, nondefault_stream):
    """The pending H2D copy must complete before its pinned storage is replaced."""
    _check_offload_staging_reuse(dtype, nondefault_stream, chunk_size=1024, sizes=(127, 4113, 63, 2053), steps=8)


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
def test_accelerator_offload_large_chunk_staging_matches_reference():
    """Exercise multi-megabyte DMA and a final short chunk on a non-default stream."""
    _check_offload_staging_reuse(
        torch.bfloat16, nondefault_stream=True, chunk_size=1_048_576, sizes=(524_287, 2_097_153, 262_143), steps=3
    )


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
def test_accelerator_offload_checkpoint_group_preflight(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer

    device = _require_native_device()
    parallel = SimpleNamespace(dp_mode="ddp")

    def build(reverse=False):
        model = torch.nn.Linear(4, 3, dtype=torch.bfloat16, device=device)
        params = list(model.parameters())
        if reverse:
            params.reverse()
        leaf = CPUOffloadAdamW([{"params": params, "cpu_offload": True}], lr=0.01)
        return model, leaf, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    source, source_leaf, optimizer = build()
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    source.weight.requires_grad_(False)
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)

    mismatch, mismatch_leaf, mismatch_optimizer = build(reverse=True)
    mismatch.weight.requires_grad_(False)
    before = [parameter.detach().clone() for parameter in mismatch.parameters()]
    with pytest.raises(RuntimeError, match="group|layout|identity"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": mismatch, "optimizer": mismatch_optimizer}, parallel_state=parallel
        )
    assert not mismatch_leaf.state
    for actual, expected in zip(mismatch.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    target, target_leaf, restored = build()
    target.weight.requires_grad_(False)
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert not target.weight.requires_grad
    for src, dst in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(dst, src, rtol=0, atol=0)
        for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(target_leaf.state[dst][key], source_leaf.state[src][key], rtol=0, atol=0)
            assert target_leaf.state[dst][key].dtype == torch.float32
            assert target_leaf.state[dst][key].device.type == "cpu"
    source.weight.requires_grad_(True)
    target.weight.requires_grad_(True)
    for src, dst in zip(source.parameters(), target.parameters(), strict=True):
        src.grad = torch.full_like(src, 0.125)
        dst.grad = src.grad.clone()
    optimizer.step()
    restored.step()
    for src, dst in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(dst, src, rtol=0, atol=0)
        torch.testing.assert_close(
            target_leaf.state[dst]["master_weight"], source_leaf.state[src]["master_weight"], rtol=0, atol=0
        )


def _check_offload_staging_reuse(dtype, nondefault_stream, chunk_size, sizes, steps):
    device = _require_native_device()
    generator = torch.Generator(device="cpu").manual_seed(734)
    initial = [torch.randn(size, dtype=dtype, generator=generator) for size in sizes]
    params = [torch.nn.Parameter(value.to(device)) for value in initial]
    references = [torch.nn.Parameter(value.float()) for value in initial]
    optimizer = CPUOffloadAdamW([{"params": params, "cpu_offload": True}], lr=1e-3, chunk_size=chunk_size)
    optimizer_ref = weakref.ref(optimizer)
    growth = []

    def track_buffer(tensor, size):
        current = optimizer_ref()
        assert current is not None
        key = (tensor.device, tensor.dtype)
        old_size = current._staging[key].numel() if key in current._staging else None
        result = CPUOffloadAdamW._buffer(current, tensor, size)
        new_size = current._staging[key].numel()
        if old_size is not None and new_size > old_size:
            growth.append((old_size, new_size))
        return result

    optimizer._buffer = track_buffer
    expected = torch.optim.AdamW(references, lr=1e-3, betas=(0.9, 0.95), weight_decay=0, foreach=False)
    runtime = get_torch_device()
    context = runtime.stream(runtime.Stream(device=device)) if nondefault_stream else contextlib.nullcontext()
    with context:
        for _ in range(steps):
            for param, reference, source in zip(params, references, initial, strict=True):
                gradient = torch.randn(source.shape, dtype=source.dtype, generator=generator)
                param.grad = gradient.to(device)
                reference.grad = gradient.float()
            optimizer.step()
            expected.step()
            for param, reference in zip(params, references, strict=True):
                state = optimizer.state[param]
                torch.testing.assert_close(state["master_weight"], reference, rtol=1e-6, atol=1e-7)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(state[key], expected.state[reference][key], rtol=1e-6, atol=1e-7)
                tolerance = {"rtol": 0, "atol": 0} if dtype is torch.bfloat16 else {"rtol": 1e-6, "atol": 1e-7}
                torch.testing.assert_close(param.detach().cpu(), reference.to(dtype), **tolerance)
    assert growth, "test inputs must exercise staging-buffer expansion"
