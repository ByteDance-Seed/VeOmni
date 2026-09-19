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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lifecycle, dispatcher, numerical, and gradient tests for the batch-invariant ATen patch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from veomni.ops.batch_invariant import patch as batch_patch
from veomni.ops.batch_invariant.support import addmm_can_fuse_bias, mean_keep_fp32_until_divide
from veomni.utils.device import IS_CUDA_AVAILABLE


class _FakeLibrary:
    instances: list[_FakeLibrary] = []
    fail_on_impl: int | None = None

    def __init__(self, namespace: str, kind: str):
        self.namespace = namespace
        self.kind = kind
        self.implementations = []
        self.destroyed = False
        self.instances.append(self)

    def impl(self, op_name, implementation, dispatch_key):
        self.implementations.append((op_name, implementation, dispatch_key))
        if len(self.implementations) == self.fail_on_impl:
            raise RuntimeError("registration failed")

    def _destroy(self):
        self.destroyed = True


@pytest.fixture
def fake_library(monkeypatch):
    batch_patch.disable_batch_invariant_mode()
    _FakeLibrary.instances.clear()
    _FakeLibrary.fail_on_impl = None

    implementations = tuple(
        (name, object()) for name in ("aten::mm", "aten::addmm", "aten::_log_softmax", "aten::mean.dim")
    )
    monkeypatch.setattr(batch_patch, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(batch_patch, "_batch_invariant_implementations", lambda: implementations)
    monkeypatch.setattr(batch_patch.torch.library, "Library", _FakeLibrary)
    monkeypatch.setattr(
        batch_patch.torch.accelerator,
        "current_accelerator",
        lambda: SimpleNamespace(type="test_accelerator"),
    )

    yield implementations

    batch_patch.disable_batch_invariant_mode()


def test_enable_is_idempotent_and_registers_all_implementations(fake_library):
    batch_patch.enable_batch_invariant_mode()
    batch_patch.enable_batch_invariant_mode()

    assert batch_patch.is_batch_invariant_mode_enabled()
    assert len(_FakeLibrary.instances) == 1
    library = _FakeLibrary.instances[0]
    assert library.namespace == "aten"
    assert library.kind == "IMPL"
    assert library.implementations == [
        (name, implementation, "TEST_ACCELERATOR") for name, implementation in fake_library
    ]

    batch_patch.disable_batch_invariant_mode()
    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert library.destroyed


def test_context_restores_state_after_exception(fake_library):
    with pytest.raises(RuntimeError, match="body failed"):
        with batch_patch.set_batch_invariant_mode():
            assert batch_patch.is_batch_invariant_mode_enabled()
            raise RuntimeError("body failed")

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert _FakeLibrary.instances[0].destroyed


def test_nested_enabled_context_reuses_outer_install(fake_library):
    with batch_patch.set_batch_invariant_mode():
        outer_library = _FakeLibrary.instances[0]
        with batch_patch.set_batch_invariant_mode(True):
            assert batch_patch.is_batch_invariant_mode_enabled()
            assert len(_FakeLibrary.instances) == 1
        assert batch_patch.is_batch_invariant_mode_enabled()
        assert not outer_library.destroyed

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert outer_library.destroyed


def test_nested_disabled_context_restores_outer_install(fake_library):
    with batch_patch.set_batch_invariant_mode(True):
        outer_library = _FakeLibrary.instances[0]
        with batch_patch.set_batch_invariant_mode(False):
            assert not batch_patch.is_batch_invariant_mode_enabled()
            assert outer_library.destroyed

        assert batch_patch.is_batch_invariant_mode_enabled()
        restored_library = _FakeLibrary.instances[-1]
        assert restored_library is not outer_library
        assert not restored_library.destroyed

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert restored_library.destroyed


def test_context_restores_manual_enable(fake_library):
    batch_patch.enable_batch_invariant_mode()
    manual_library = _FakeLibrary.instances[0]

    with batch_patch.set_batch_invariant_mode(False):
        assert not batch_patch.is_batch_invariant_mode_enabled()
        assert manual_library.destroyed

    assert batch_patch.is_batch_invariant_mode_enabled()
    assert len(_FakeLibrary.instances) == 2


def test_context_is_noop_without_cuda(fake_library, monkeypatch):
    monkeypatch.setattr(batch_patch, "IS_CUDA_AVAILABLE", False)

    with batch_patch.set_batch_invariant_mode(True):
        assert not batch_patch.is_batch_invariant_mode_enabled()

    assert _FakeLibrary.instances == []


def test_failed_registration_destroys_partial_library(fake_library):
    _FakeLibrary.fail_on_impl = 3

    with pytest.raises(RuntimeError, match="registration failed"):
        batch_patch.enable_batch_invariant_mode()

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert len(_FakeLibrary.instances) == 1
    assert _FakeLibrary.instances[0].destroyed


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="batch-invariant handlers require CUDA + Triton")
@pytest.mark.parametrize("op_name", ("mm", "addmm", "log_softmax", "mean"))
def test_real_handler_matches_torch_output_gradient_and_dispatcher(op_name, monkeypatch):
    """Exercise each real handler through its public ATen-dispatched torch API."""
    batch_patch.disable_batch_invariant_mode()
    torch.manual_seed(17)
    calls: list[str] = []
    implementations = batch_patch._batch_invariant_implementations()

    def track(name, implementation):
        def tracked(*args, **kwargs):
            calls.append(name)
            return implementation(*args, **kwargs)

        return tracked

    monkeypatch.setattr(
        batch_patch,
        "_batch_invariant_implementations",
        lambda: tuple((name, track(name, implementation)) for name, implementation in implementations),
    )

    if op_name in {"mm", "addmm"}:
        input_bases = (
            torch.randn(37, 29, device="cuda", dtype=torch.bfloat16),
            torch.randn(29, 23, device="cuda", dtype=torch.bfloat16),
        )
        if op_name == "addmm":
            input_bases = (torch.randn(23, device="cuda", dtype=torch.bfloat16), *input_bases)
        operation = torch.mm if op_name == "mm" else torch.addmm
    elif op_name == "log_softmax":
        input_bases = (torch.randn(7, 37, device="cuda", dtype=torch.float32),)

        def operation(value):
            return torch.log_softmax(value, dim=-1)

    else:
        input_bases = (torch.randn(3, 11, 7, device="cuda", dtype=torch.float32),)

        def operation(value):
            return torch.mean(value, dim=1, keepdim=True)

    expected_inputs = tuple(value.detach().clone().requires_grad_(True) for value in input_bases)
    actual_inputs = tuple(value.detach().clone().requires_grad_(True) for value in input_bases)
    expected = operation(*expected_inputs)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)

    with batch_patch.set_batch_invariant_mode():
        assert batch_patch.is_batch_invariant_mode_enabled()
        actual = operation(*actual_inputs)
        actual.backward(grad_output)
    assert not batch_patch.is_batch_invariant_mode_enabled()

    expected_dispatch = {
        "mm": "aten::mm",
        "addmm": "aten::addmm",
        "log_softmax": "aten::_log_softmax",
        "mean": "aten::mean.dim",
    }
    assert expected_dispatch[op_name] in calls
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="batch-invariant handlers require CUDA + Triton")
@pytest.mark.parametrize("op_name", ("mm", "addmm", "log_softmax", "mean"))
def test_real_handler_is_invariant_to_batch_partition(op_name):
    """A sample and its input gradient are bit-identical across batch partitions."""
    batch_patch.disable_batch_invariant_mode()
    torch.manual_seed(23)

    if op_name in {"mm", "addmm"}:
        values = torch.randn(7, 29, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(29, 23, device="cuda", dtype=torch.bfloat16)
        if op_name == "mm":

            def operation(value):
                return torch.mm(value, weight)

        else:
            bias = torch.randn(23, device="cuda", dtype=torch.bfloat16)

            def operation(value):
                return torch.addmm(bias, value, weight)

    elif op_name == "log_softmax":
        values = torch.randn(7, 37, device="cuda", dtype=torch.float32)

        def operation(value):
            return torch.log_softmax(value, dim=-1)

    else:
        values = torch.randn(7, 11, 13, device="cuda", dtype=torch.float32)

        def operation(value):
            return torch.mean(value, dim=1, keepdim=True)

    with batch_patch.set_batch_invariant_mode():
        joint_input = values.detach().clone().requires_grad_(True)
        joint_output = operation(joint_input)
        grad_output = torch.randn_like(joint_output)
        joint_gradient = torch.autograd.grad(joint_output, joint_input, grad_output)[0]

        partition_outputs = []
        partition_gradients = []
        for value, gradient in zip(values.split((2, 5)), grad_output.split((2, 5)), strict=True):
            partition_input = value.detach().clone().requires_grad_(True)
            partition_output = operation(partition_input)
            partition_outputs.append(partition_output)
            partition_gradients.append(torch.autograd.grad(partition_output, partition_input, gradient)[0])

    assert torch.equal(joint_output, torch.cat(partition_outputs))
    assert torch.equal(joint_gradient, torch.cat(partition_gradients))


def test_mean_keep_fp32_until_divide_avoids_fp16_overflow():
    values = torch.ones(256, 256, dtype=torch.float16)
    overflowed = torch.sum(values, dim=(0, 1), dtype=torch.float32).to(torch.float16) / values.numel()
    assert not torch.isfinite(overflowed)
    actual = mean_keep_fp32_until_divide(values, (0, 1))
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, torch.ones((), dtype=torch.float16))


def test_mean_batch_invariant_single_dim_forwards_dtype(monkeypatch):
    """Single-dim mean must honor an explicit dtype, not only the multi-dim path."""
    from veomni.ops.batch_invariant import triton as module

    def fake_mean_dim(input, dim, keepdim=False, dtype=None):
        return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)

    monkeypatch.setattr(module, "mean_dim", fake_mean_dim)
    values = torch.ones(4, 8, dtype=torch.float16)
    actual = module.mean_batch_invariant(values, (1,), dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.ones(4, dtype=torch.float32))


@pytest.mark.parametrize(
    ("bias_factory", "n", "beta", "alpha", "expected"),
    (
        (lambda: torch.randn(4), 4, 1, 1, True),
        (lambda: None, 4, 1, 1, True),
        (lambda: torch.randn(1), 4, 1, 1, False),
        (lambda: torch.randn(4, 1).expand(4, 4)[0], 4, 1, 1, False),
        (lambda: torch.randn(4), 4, 0, 1, False),
        (lambda: torch.randn(4), 4, 1, 2, False),
        (lambda: torch.randn(2, 4), 4, 1, 1, False),
    ),
    ids=("contig-1d", "no-bias", "broadcast-len1", "nonunit-stride", "beta0", "alpha2", "2d-bias"),
)
def test_addmm_can_fuse_bias_rejects_unsupported_pairs(bias_factory, n, beta, alpha, expected):
    bias = bias_factory()
    assert addmm_can_fuse_bias(bias, n, beta=beta, alpha=alpha) is expected


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="addmm fallback uses the Triton mm path")
def test_addmm_falls_back_for_alpha_and_broadcast_bias():
    from veomni.ops.batch_invariant.triton import addmm_batch_invariant, mm_batch_invariant

    torch.manual_seed(3)
    a = torch.randn(5, 7, device="cuda", dtype=torch.float32)
    b = torch.randn(7, 4, device="cuda", dtype=torch.float32)
    bias = torch.tensor([2.0], device="cuda")
    actual = addmm_batch_invariant(bias, a, b, beta=0.5, alpha=2)
    expected = 2 * mm_batch_invariant(a, b) + 0.5 * bias
    torch.testing.assert_close(actual, expected)


def test_addmm_beta_zero_skips_nan_bias(monkeypatch):
    """beta=0 must not read bias. Do not use a patched aten::addmm as the oracle."""
    from veomni.ops.batch_invariant import triton as module

    monkeypatch.setattr(module, "mm_batch_invariant", lambda left, right: left @ right)
    torch.manual_seed(4)
    a = torch.randn(5, 7)
    b = torch.randn(7, 4)
    bias = torch.full((4,), float("nan"))
    actual = module.addmm_batch_invariant(bias, a, b, beta=0)
    expected = torch.addmm(bias, a, b, beta=0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, a @ b)
