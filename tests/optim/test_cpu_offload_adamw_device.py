"""Native staging coverage, separate from the CPU-only optimizer suite."""

import contextlib
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, ModelState, OptimizerState
from veomni.checkpoint.layout import optimizer_dir, weights_dir
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
    device = _require_native_device()
    parallel = SimpleNamespace(dp_mode="ddp")

    def build(reverse=False):
        model = torch.nn.Linear(4, 3, dtype=torch.bfloat16, device=device)
        params = list(model.parameters())
        if reverse:
            params.reverse()
        optimizer = CPUOffloadAdamW([{"params": params, "cpu_offload": True}], lr=0.01)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        return model, optimizer

    source_model, source = build()
    dcp.save({"model": ModelState(source_model, parallel_state=parallel)}, checkpoint_id=weights_dir(str(tmp_path)))
    dcp.save(
        {"optimizer": OptimizerState(source_model, source, parallel_state=parallel)},
        checkpoint_id=optimizer_dir(str(tmp_path)),
    )
    optimizer_metadata = (tmp_path / "model" / "optimizer" / ".metadata").read_bytes()
    source_model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen precision-sensitive optimizer state"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": source_model, "optimizer": source}, parallel_state=parallel
        )
    assert (tmp_path / "model" / "optimizer" / ".metadata").read_bytes() == optimizer_metadata
    source_model.weight.requires_grad_(True)
    target_model, target = build(reverse=True)
    weights = [parameter.detach().clone() for parameter in target_model.parameters()]
    states = [
        {key: value.clone() for key, value in target.state[parameter].items()}
        for parameter in target_model.parameters()
    ]
    with pytest.raises(RuntimeError, match="parameter groups"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": target_model, "optimizer": target}, parallel_state=parallel
        )
    for index, parameter in enumerate(target_model.parameters()):
        torch.testing.assert_close(parameter, weights[index], rtol=0, atol=0)
        for key, value in states[index].items():
            torch.testing.assert_close(target.state[parameter][key], value, rtol=0, atol=0)

    restored_model, restored = build()
    restored_model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="restore frozen precision-sensitive optimizer parameter"):
        DistributedCheckpointer.load(
            str(tmp_path), {"model": restored_model, "optimizer": restored}, parallel_state=parallel
        )
    restored_model.weight.requires_grad_(True)
    before_states = [
        {key: value.clone() for key, value in restored.state[parameter].items()}
        for parameter in restored_model.parameters()
    ]

    def freeze_weight(module, _incompatible):
        module.weight.requires_grad_(False)

    hook = restored_model.register_load_state_dict_post_hook(freeze_weight)
    try:
        with pytest.raises(RuntimeError, match="restore frozen precision-sensitive optimizer parameter"):
            DistributedCheckpointer.load(
                str(tmp_path), {"model": restored_model, "optimizer": restored}, parallel_state=parallel
            )
    finally:
        hook.remove()
    for index, parameter in enumerate(restored_model.parameters()):
        for key, value in before_states[index].items():
            torch.testing.assert_close(restored.state[parameter][key], value, rtol=0, atol=0)
    restored_model.weight.requires_grad_(True)
    DistributedCheckpointer.load(
        str(tmp_path), {"model": restored_model, "optimizer": restored}, parallel_state=parallel
    )
    for original, recovered in zip(source_model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(original, recovered, rtol=0, atol=0)
        for key in source.state[original]:
            torch.testing.assert_close(source.state[original][key], restored.state[recovered][key], rtol=0, atol=0)
        original.grad = torch.full_like(original, 0.5)
        recovered.grad = torch.full_like(recovered, 0.5)
    source.step()
    restored.step()
    for original, recovered in zip(source_model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(original, recovered, rtol=0, atol=0)


def _check_offload_staging_reuse(dtype, nondefault_stream, chunk_size, sizes, steps):
    device = _require_native_device()
    generator = torch.Generator(device="cpu").manual_seed(734)
    initial = [torch.randn(size, dtype=dtype, generator=generator) for size in sizes]
    params = [torch.nn.Parameter(value.to(device)) for value in initial]
    references = [torch.nn.Parameter(value.float()) for value in initial]
    optimizer = CPUOffloadAdamW([{"params": params, "cpu_offload": True}], lr=1e-3, chunk_size=chunk_size)
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
