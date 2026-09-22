"""CPU contract checks for the NPU wrapper; native parity is a worker gate."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


def load_wrapper():
    def rms(x, weight, eps):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        return (normalized.to(x.dtype) * weight,)

    def swiglu(x, dim):
        gate, value = x.chunk(2, dim=dim)
        return F.silu(gate) * value

    fake = types.SimpleNamespace(npu_rms_norm=rms, npu_swiglu=swiglu)
    path = Path(__file__).parents[2] / "veomni/ops/kernels/gated_delta_rule/npu_rms_norm_gated.py"
    spec = importlib.util.spec_from_file_location("_gated_norm_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch_npu": fake}):
        spec.loader.exec_module(module)
    return module.NPUFusedRMSNormGated


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sigmoid_reference_and_gradients(dtype):
    torch.manual_seed(73)
    module = load_wrapper()(128, dtype=dtype, activation="sigmoid")
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.25, 1.75, 128).to(dtype))
    x = torch.randn(7, 128, dtype=dtype).requires_grad_()
    z = torch.linspace(-20, 20, 896).reshape(7, 128).to(dtype).requires_grad_()
    r = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + module.variance_epsilon)
    expected = (module.weight * r.to(dtype) * z.float().sigmoid()).to(dtype)
    actual = module(x, z)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    dy = torch.randn_like(actual)
    got = torch.autograd.grad(actual, (x, z, module.weight), dy)
    ref = torch.autograd.grad(expected, (x, z, module.weight), dy)
    for a, b in zip(got, ref, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zero_gate_is_half_normalized_and_silu_unchanged():
    cls = load_wrapper()
    x = torch.ones(2, 128)
    z = torch.zeros_like(x)
    torch.testing.assert_close(cls(128, activation="sigmoid")(x, z), x * 0.5 / (1 + 1e-6) ** 0.5)
    torch.testing.assert_close(cls(128)(x, z), torch.zeros_like(x))
    with pytest.raises(ValueError, match="activation"):
        cls(128, activation="relu")
    with pytest.raises(ValueError, match="gate"):
        cls(128, activation="sigmoid")(x)
