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

from scripts.ci.accelerator.launch_plan import LaunchSpec
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


def test_native_npu_fp32_wgrad_matches_cpu_reference():
    if not IS_NPU_AVAILABLE:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires a native NPU")
        pytest.skip("native NPU is unavailable")
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(20261029)
    counts = [13, 0, 17]
    x = torch.randn(30, 256, dtype=torch.bfloat16, device="npu", requires_grad=True)
    weight = torch.randn(3, 256, 32, dtype=torch.float32, device="npu", requires_grad=True)
    dy = torch.randn(30, 32, dtype=torch.bfloat16, device="npu")
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    output = GmmFunction.apply(x, weight, groups, True)
    output.backward(dy)
    torch.npu.synchronize()
    expected = _reference(x.detach().cpu(), dy.cpu(), counts)
    assert weight.grad.dtype == torch.float32
    torch.testing.assert_close(weight.grad.cpu(), expected, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(weight.grad[1]).item() == 0
