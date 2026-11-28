# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward and backward numerical contract for the functional MXFP8 projection."""

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OpsSelection, VeomniOp
from veomni.ops.torchao import torchao_mxfp8_linear
from veomni.utils import device


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _rounded(operand):
    # Independent E4M3 round trip. Round the block scale up to a power of two
    # so that its largest magnitude fits in the E4M3 range.
    blocks = operand.float().reshape(-1, 32)
    peak = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(peak > 0, (peak / 448).log2().ceil().exp2(), 1.0)
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(operand).bfloat16()


def _reference(x, weight, bias, dy):
    x_rows, dy_rows = x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1])
    pad = (-x_rows.shape[0]) % 32
    xp, gp = F.pad(x_rows, (0, 0, 0, pad)), F.pad(dy_rows, (0, 0, 0, pad))
    output = (_rounded(xp) @ _rounded(weight).T)[: x_rows.shape[0]].reshape_as(dy) + bias.bfloat16()
    dx = (_rounded(gp) @ _rounded(weight.T).T)[: x_rows.shape[0]].reshape_as(x).to(x.dtype)
    dw = (_rounded(gp.T) @ _rounded(xp.T).T).to(weight.dtype)
    return output, dx, dw, dy_rows.sum(0).to(bias.dtype)


def _check_projection(tokens, weight_dtype, target_device, *, emulated, autocast_dtype=None):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(1, tokens, 64, dtype=torch.float32, generator=generator)
    weight = torch.randn(128, 64, dtype=weight_dtype, generator=generator)
    bias = torch.randn(128, generator=generator)
    dy = torch.randn(1, tokens, 128, dtype=torch.bfloat16, generator=generator)
    expected = _reference(x.bfloat16() if autocast_dtype else x, weight, bias, dy)

    layer = torch.nn.Linear(64, 128, device="meta")
    layer.weight = torch.nn.Parameter(weight.to(target_device))
    layer.bias = torch.nn.Parameter(bias.to(target_device))
    x = x.to(target_device).requires_grad_()
    selection = OpsSelection(name="torchao", kwargs={"dtype": "mxfp8", "emulated": emulated})
    op = VeomniOp("linear", "standard", selection)
    with torch.autocast(x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        output = op(layer, x)
    assert output.dtype == (autocast_dtype or x.dtype)
    output.backward(dy.to(target_device))
    for actual, reference in zip((output, x.grad, layer.weight.grad, layer.bias.grad), expected, strict=True):
        # Native and CPU emulated reductions may differ by one BF16 spacing.
        torch.testing.assert_close(
            actual.detach().cpu(),
            reference.to(actual.dtype),
            rtol=0 if emulated else torch.finfo(torch.bfloat16).eps,
            atol=0 if emulated else 1e-5,
        )


@pytest.mark.parametrize(
    ("tokens", "weight_dtype", "autocast_dtype"),
    [
        (1, torch.float32, torch.bfloat16),
        (3, torch.float32, torch.float16),
        (32, torch.bfloat16, None),
        (65, torch.float32, None),
    ],
)
def test_emulated_mxfp8_matches_independent_oracle(tokens, weight_dtype, autocast_dtype):
    pytest.importorskip("torchao")
    _check_projection(tokens, weight_dtype, "cpu", emulated=True, autocast_dtype=autocast_dtype)


@pytest.mark.parametrize(
    ("register", "hook"),
    [
        ("register_forward_hook", lambda module, args, output: 2 * output),
        ("register_forward_pre_hook", lambda module, args: (2 * args[0],)),
        ("register_full_backward_hook", lambda module, grad_input, grad_output: (2 * grad_input[0],)),
        ("register_full_backward_pre_hook", lambda module, grad_output: (2 * grad_output[0],)),
    ],
)
def test_mxfp8_rejects_projection_hooks(register, hook):
    from veomni.ops.kernels.linear.standard import torchao as linear_torchao

    layer = torch.nn.Linear(32, 32)
    getattr(layer, register)(hook)
    with pytest.raises(ValueError, match="MXFP8.*projection.*hooks"):
        linear_torchao.wrapper(layer, torch.ones(1, 32), dtype="mxfp8", emulated=True)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
def test_native_mxfp8_matches_independent_oracle():
    if not device.IS_CUDA_AVAILABLE or torch.version.hip is not None or device.get_gpu_compute_capability() < 100:
        pytest.skip("Requires native MXFP8 on NVIDIA SM100+.")
    pytest.importorskip("torchao")
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as trace:
        _check_projection(33, torch.float32, device.get_device_type(), emulated=False)
    assert sum(event.count for event in trace.key_averages() if event.key == "aten::_scaled_mm") == 3


def test_mxfp8_handles_empty_input():
    pytest.importorskip("torchao")
    weight = torch.randn(128, 64, requires_grad=True)
    bias = torch.randn(128, requires_grad=True)
    x = torch.empty(2, 0, 64, requires_grad=True)
    output = torchao_mxfp8_linear(x, weight, bias, emulated=True)
    assert output.dtype == torch.bfloat16 and output.shape == (2, 0, 128)
    output.sum().backward()
    for operand in (x, weight, bias):
        torch.testing.assert_close(operand.grad, torch.zeros_like(operand))


def test_mxfp8_aot_autograd_matches_eager(monkeypatch):
    pytest.importorskip("torchao")
    from veomni.ops import torchao as torchao_ops

    # The first compiled call must also pass the package-version check.
    monkeypatch.setattr(torchao_ops, "_CHECKED_MXFP8_DEVICES", set())
    compiled = torch.compile(torchao_mxfp8_linear, backend="aot_eager", fullgraph=True)
    x = torch.randn(17, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 64, requires_grad=True)
    expected_x, expected_w = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    output = compiled(x, weight, emulated=True)
    expected = torchao_mxfp8_linear(expected_x, expected_w, emulated=True)
    dy = torch.randn_like(output)
    output.backward(dy)
    expected.backward(dy)
    for actual, reference in ((output, expected), (x.grad, expected_x.grad), (weight.grad, expected_w.grad)):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_mxfp8_rejects_unsupported_shapes_and_device():
    x = torch.zeros(3, 64)
    with pytest.raises(ValueError, match="divisible by 32"):
        torchao_mxfp8_linear(x, torch.zeros(48, 64), emulated=True)
    with pytest.raises(ValueError, match="input features"):
        torchao_mxfp8_linear(x, torch.zeros(32, 32), emulated=True)
    pytest.importorskip("torchao")
    with pytest.raises(RuntimeError, match="SM100"):
        torchao_mxfp8_linear(x, torch.zeros(32, 64))
