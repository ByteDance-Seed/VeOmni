"""Native staging coverage, separate from the CPU-only optimizer suite."""

import os

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
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
