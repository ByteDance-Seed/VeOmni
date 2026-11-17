# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU expert GEMM backward precision, including the BF16 default path."""

import os
import sys
import types

import pytest
import torch
import torch.utils.checkpoint

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from veomni.ops.kernels.moe_experts.standard.npu import GmmFunction
from veomni.utils.device import IS_NPU_AVAILABLE


def _grouped(xs, weights, *, group_list, group_type, **kwargs):
    x, weight = xs[0], weights[0]
    assert x.dtype == weight.dtype
    out, offset = [], 0
    for expert, count in enumerate(group_list):
        count = int(count)
        if group_type == 0:
            out.append(x[offset : offset + count] @ weight[expert])
        else:
            assert group_type == 2
            out.append(x[:, offset : offset + count] @ weight[offset : offset + count])
        offset += count
    return [torch.cat(out) if group_type == 0 else torch.stack(out)]


def _reference(x, dy, counts):
    start, grads = 0, []
    for count in counts:
        grads.append(x[start : start + count].float().T @ dy[start : start + count].float())
        start += count
    return torch.stack(grads)


@pytest.fixture
def fake_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_grouped_matmul=_grouped))


def _cpu_inputs(counts=(6, 7)):
    torch.manual_seed(714)
    return (
        torch.randn(sum(counts), 11, dtype=torch.bfloat16),
        torch.randn(len(counts), 11, 9),
        torch.randn(sum(counts), 9, dtype=torch.bfloat16),
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("enabled", [False, True])
def test_npu_wrapper_accepts_quantization_options_and_forwards_fp32_wgrad(monkeypatch, enabled):
    from veomni.ops.kernels.moe_experts.standard import npu

    captured = {}

    def forward(*args, **kwargs):
        captured.update(kwargs)
        return args[3]

    monkeypatch.setattr(npu, "npu_fused_moe_forward", forward)
    tensor = torch.ones(1)
    result = npu.wrapper(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        num_experts=1,
        block_size=64,
        expert_dtype="fp4",
        fp32_wgrad=enabled,
    )
    assert result is tensor
    assert captured["fp32_wgrad"] is enabled


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_fp32_wgrad_keeps_bf16_forward_and_input_gradient(fake_npu):
    x, master, dy = _cpu_inputs()
    counts = [6, 7]
    inp, param = x.clone().requires_grad_(True), master.clone().requires_grad_(True)
    output = GmmFunction.apply(inp, param, counts, True)
    output.backward(dy)
    assert torch.equal(output.detach(), _grouped([x], [master.bfloat16()], group_list=counts, group_type=0)[0])
    expected_dx = _grouped([dy], [master.bfloat16().transpose(1, 2)], group_list=counts, group_type=0)[0]
    assert torch.equal(inp.grad, expected_dx)
    assert param.grad.dtype == torch.float32
    assert torch.equal(param.grad, _reference(x, dy, counts))
    assert not torch.equal(param.grad, param.grad.bfloat16().float())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("saved_hooks", [False, True])
def test_fp32_weight_rematerialization_preserves_gradients_and_hook_fallback(fake_npu, saved_hooks):
    from contextlib import nullcontext

    x, master, dy = _cpu_inputs()
    reference = []
    for rematerialize in (False, True):
        inp, param = x.clone().requires_grad_(True), master.clone().requires_grad_(True)
        hooks = (
            torch.autograd.graph.saved_tensors_hooks(lambda t: t.clone(), lambda t: t)
            if saved_hooks
            else nullcontext()
        )
        with hooks:
            output = GmmFunction.apply(inp, param, [6, 7], True, rematerialize)
        saved_dtype = output.grad_fn.saved_tensors[1].dtype
        assert saved_dtype == (torch.float32 if rematerialize and not saved_hooks else torch.bfloat16)
        output.backward(dy)
        reference.append((output.detach(), inp.grad, param.grad))
    for expected, actual in zip(*reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_fp32_weight_rematerialization_falls_back_if_hook_query_fails(fake_npu, monkeypatch):
    def fail(_):
        raise RuntimeError("unsupported hook query")

    monkeypatch.setattr(torch._C._autograd, "_top_saved_tensors_default_hooks", fail)
    x, master, dy = _cpu_inputs()
    param = master.requires_grad_(True)
    output = GmmFunction.apply(x, param, [6, 7], True, True)
    assert output.grad_fn.saved_tensors[1].dtype == torch.bfloat16
    output.backward(dy)
    assert torch.equal(param.grad, _reference(x, dy, [6, 7]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("reentrant", [False, True])
def test_fp32_weight_rematerialization_with_transposed_weight_and_checkpoint(fake_npu, reentrant):
    x, master, dy = _cpu_inputs()
    results = []
    for rematerialize in (False, True):
        inp = x.clone().requires_grad_(True)
        param = master.transpose(1, 2).contiguous().requires_grad_(True)
        output = torch.utils.checkpoint.checkpoint(
            lambda a, b, remat=rematerialize: GmmFunction.apply(a, b.transpose(1, 2), [6, 7], True, remat),
            inp,
            param,
            use_reentrant=reentrant,
        )
        output.backward(dy)
        results.append((output.detach(), inp.grad, param.grad))
    for expected, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_bf16_default_and_cast_before_autograd_preserve_legacy_rounding(fake_npu):
    x, master, dy = _cpu_inputs()
    counts = [6, 7]
    param = master.clone().requires_grad_(True)
    GmmFunction.apply(x, param.to(torch.bfloat16), counts).backward(dy)
    assert torch.equal(param.grad, param.grad.bfloat16().float())
    assert not torch.equal(param.grad, _reference(x, dy, counts))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("reentrant", [True, False])
def test_checkpoint_recomputation_keeps_fp32_weight_gradient(fake_npu, reentrant):
    x, master, dy = _cpu_inputs()
    counts = [6, 7]
    inp, param = x.clone().requires_grad_(True), master.clone().requires_grad_(True)
    out = torch.utils.checkpoint.checkpoint(
        lambda a, b: GmmFunction.apply(a, b, counts, True), inp, param, use_reentrant=reentrant
    )
    out.backward(dy)
    assert torch.equal(param.grad, _reference(x, dy, counts))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("counts", [[4, 0, 4], [0, 0, 0]])
def test_empty_experts_keep_zero_fp32_gradient(fake_npu, counts):
    x, master, dy = _cpu_inputs(counts)
    param = master.requires_grad_(True)
    GmmFunction.apply(x, param, counts, True).backward(dy)
    assert param.grad.dtype == torch.float32
    assert torch.equal(param.grad, _reference(x, dy, counts))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_bad_compute_or_master_dtype_is_rejected(fake_npu):
    x, master, _ = _cpu_inputs()
    with pytest.raises(ValueError, match="FP32 expert wgrad"):
        GmmFunction.apply(x.float(), master, [6, 7], True)
    with pytest.raises(ValueError, match="FP32 expert wgrad"):
        GmmFunction.apply(x, master.bfloat16(), [6, 7], True)


@pytest.mark.parametrize("rematerialize", [False, True])
def test_native_npu_fp32_wgrad_matches_cpu_reference(rematerialize):
    capabilities = runner_capabilities() if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    required = os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1"
    if capabilities is not None and capabilities.backend != "npu":
        if required:
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires an NPU CI assignment")
        pytest.skip("requires an NPU CI assignment")
    npu = getattr(torch, "npu", None)
    if (capabilities is not None and not capabilities.devices) or not (
        IS_NPU_AVAILABLE and npu is not None and npu.is_available() and npu.device_count() >= 1
    ):
        if required or capabilities is not None:
            pytest.fail("Assigned or required NPU runtime must provide a usable device")
        pytest.skip("native NPU is unavailable")
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(20261029)
    counts = [13, 0, 17]
    x = torch.randn(30, 256, dtype=torch.bfloat16, device="npu", requires_grad=True)
    weight = torch.randn(3, 32, 256, dtype=torch.float32, device="npu", requires_grad=True).transpose(1, 2)
    weight.retain_grad()
    dy = torch.randn(30, 32, dtype=torch.bfloat16, device="npu")
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    baseline_x = x.detach().clone().requires_grad_(True)
    baseline_weight = weight.detach().clone().requires_grad_(True)
    baseline_output = GmmFunction.apply(baseline_x, baseline_weight, groups, True, False)
    baseline_output.backward(dy)
    output = GmmFunction.apply(x, weight, groups, True, rematerialize)
    output.backward(dy)
    torch.npu.synchronize()
    torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, baseline_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, baseline_weight.grad, rtol=0, atol=0)
    expected = _reference(x.detach().cpu(), dy.cpu(), counts)
    assert weight.grad.dtype == torch.float32
    torch.testing.assert_close(weight.grad.cpu(), expected, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(weight.grad[1]).item() == 0
