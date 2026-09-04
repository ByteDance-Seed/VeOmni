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

from veomni.kernels._kernels.moe_experts.standard import npu as npu_moe


_CLAMPED_SWIGLU_MODULE = "veomni.kernels._kernels.moe_experts.shared.npu_clamped_swiglu"


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
