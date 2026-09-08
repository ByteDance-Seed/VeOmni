# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""NPU clamped SwiGLU dispatch: Triton-Ascend when present, eager otherwise."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from veomni.ops.kernels.moe_experts.standard import npu as npu_moe
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


_CLAMPED_SWIGLU_MODULE = "veomni.ops.kernels.moe_experts.shared.npu_clamped_swiglu"


def _eager_clamped_swiglu(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def test_npu_swiglu_dispatches_by_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.empty((2, 16), dtype=torch.bfloat16)
    clamped_output = object()
    unclamped_output = object()
    calls: list[tuple[str, bool, float | int]] = []

    def fake_clamped_swiglu(actual_x, limit):
        calls.append(("clamped", actual_x is x, limit))
        return clamped_output

    def fake_npu_swiglu(actual_x, *, dim):
        calls.append(("unclamped", actual_x is x, dim))
        return unclamped_output

    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.npu_swiglu = fake_npu_swiglu
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(npu_moe, "_clamped_swiglu", fake_clamped_swiglu)

    assert npu_moe._swiglu(x, 7.0) is clamped_output
    assert npu_moe._swiglu(x, None) is unclamped_output
    assert calls == [("clamped", True, 7.0), ("unclamped", True, -1)]


def test_npu_clamped_swiglu_missing_triton_uses_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(npu_moe, "_is_triton_ascend_available", lambda: False)
    source = torch.tensor([[8.0, -7.0, 9.0, -9.0]], requires_grad=True)
    expected_input = source.detach().clone().requires_grad_()

    actual = npu_moe._clamped_swiglu(source, 7.0)
    expected_gate, expected_up = expected_input.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(expected_gate.clamp(max=7.0)) * expected_up.clamp(min=-7.0, max=7.0)
    grad_output = torch.tensor([[0.25, -0.5]])
    actual.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source.grad, expected_input.grad, rtol=0, atol=0)


def test_npu_clamped_swiglu_requires_ascend_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_triton = ModuleType("triton")
    fake_triton.__path__ = []
    fake_triton_c = ModuleType("triton._C")
    fake_triton_c.libtriton = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton._C", fake_triton_c)

    assert not npu_moe._is_triton_ascend_available()
    fake_triton_c.libtriton.ascend = object()
    assert npu_moe._is_triton_ascend_available()


def test_npu_clamped_swiglu_dispatches_to_ascend_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.empty((1, 2))
    output = object()
    fake_kernel = ModuleType(_CLAMPED_SWIGLU_MODULE)
    fake_kernel.npu_triton_clamped_swiglu = lambda actual_x, limit: output if actual_x is x and limit == 7.0 else None
    monkeypatch.setitem(sys.modules, _CLAMPED_SWIGLU_MODULE, fake_kernel)
    monkeypatch.setattr(npu_moe, "_is_triton_ascend_available", lambda: True)

    assert npu_moe._clamped_swiglu(x, 7.0) is output


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU kernels require torch_npu")
class TestNPUClampedSwiGLUKernel:
    @pytest.fixture(autouse=True)
    def require_triton_ascend(self):
        try:
            from triton._C import libtriton
        except ImportError:
            pytest.skip("clamped SwiGLU requires triton-ascend")
        if not hasattr(libtriton, "ascend"):
            pytest.skip("clamped SwiGLU requires the Triton Ascend backend")

    @pytest.mark.parametrize("shape", [(3, 34), (2, 3, 256)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_backward_matches_eager(self, shape, dtype):
        from veomni.ops.kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        torch.manual_seed(17)
        source = torch.randn(shape, dtype=torch.float32).mul_(12)
        actual_input = source.to(device=get_device_type(), dtype=dtype).requires_grad_()
        expected_input = source.to(device=get_device_type(), dtype=dtype).requires_grad_()

        actual = npu_triton_clamped_swiglu(actual_input, 7.0)
        expected = _eager_clamped_swiglu(expected_input, 7.0)
        grad_output = torch.randn_like(actual)
        actual.backward(grad_output)
        expected.backward(grad_output)

        tolerance = 5e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual.float(), expected.float(), rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            actual_input.grad.float(), expected_input.grad.float(), rtol=tolerance, atol=tolerance
        )

        hidden_size = actual_input.shape[-1] // 2
        gate, up = actual_input.detach().split(hidden_size, dim=-1)
        gate_grad, up_grad = actual_input.grad.split(hidden_size, dim=-1)
        assert torch.count_nonzero(gate_grad[gate > 7.0]) == 0
        assert torch.count_nonzero(up_grad[(up < -7.0) | (up > 7.0)]) == 0

    def test_clamp_boundaries_match_eager_gradients(self):
        from veomni.ops.kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        values = torch.tensor([[-8.0, -7.0, 7.0, 8.0, -8.0, -7.0, 7.0, 8.0]], device=get_device_type())
        actual_input = values.clone().requires_grad_()
        expected_input = values.clone().requires_grad_()

        actual = npu_triton_clamped_swiglu(actual_input, 7.0)
        expected = _eager_clamped_swiglu(expected_input, 7.0)
        actual.sum().backward()
        expected.sum().backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-5)

    def test_empty_batch_preserves_shape_and_backward(self):
        from veomni.ops.kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        x = torch.empty((0, 30), device=get_device_type(), dtype=torch.bfloat16, requires_grad=True)

        output = npu_triton_clamped_swiglu(x, 7.0)
        output.sum().backward()

        assert output.shape == (0, 15)
        assert x.grad.shape == x.shape

    def test_rejects_odd_last_dimension(self):
        from veomni.ops.kernels.moe_experts.shared.npu_clamped_swiglu import npu_triton_clamped_swiglu

        x = torch.empty((2, 15), device=get_device_type(), dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="even last dimension"):
            npu_triton_clamped_swiglu(x, 7.0)
