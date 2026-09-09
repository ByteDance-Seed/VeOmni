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

"""CPU tests for OpEntry / VeomniOp."""

from __future__ import annotations

from inspect import Parameter, signature

import pytest
import torch
from torch import Tensor

from veomni.ops import OP_REGISTRY, VeomniOp, register_op, resolve_op
from veomni.ops.platform import (
    ANY_DEVICE,
    GpuKernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
    NvidiaGpuPlatform,
)
from veomni.ops.registry import OpEntry, SavedState


_GPU = GpuKernelRequirement()
_TEST_DESCRIPTION = "Test operation"


@pytest.fixture
def isolated_entries():
    saved_entries = dict(OP_REGISTRY._entries)
    saved_intern = dict(VeomniOp._intern)
    OP_REGISTRY._entries.clear()
    VeomniOp._intern.clear()
    yield
    OP_REGISTRY._entries.clear()
    OP_REGISTRY._entries.update(saved_entries)
    VeomniOp._intern.clear()
    VeomniOp._intern.update(saved_intern)


def _add_forward(x: Tensor, y: Tensor, *, scale: float) -> tuple[Tensor, SavedState]:
    out = (x + y) * scale
    return out, SavedState((x, y), metadata=scale)


def _add_backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    scale = saved.metadata
    return grad_output * scale, grad_output * scale


def _pair_forward(x: Tensor, y: Tensor) -> tuple[tuple[Tensor, Tensor], SavedState]:
    return (x + y, x * y), SavedState((x, y))


def _pair_backward(grad_output: tuple[Tensor, Tensor], saved: SavedState) -> tuple[Tensor | None, ...]:
    x, y = saved.tensors
    grad_sum, grad_prod = grad_output
    return grad_sum + y * grad_prod, grad_sum + x * grad_prod


def _optional_forward(x: Tensor, y: Tensor | None = None) -> tuple[Tensor, SavedState]:
    has_y = y is not None
    if y is None:
        y = torch.zeros_like(x)
    return x + y, SavedState((x, y), has_y)


def _optional_backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    return grad_output, grad_output if saved.metadata else None


@pytest.mark.usefixtures("isolated_entries")
class TestOpEntryValidation:
    @pytest.mark.parametrize("target", (OpEntry, register_op))
    def test_description_is_required_keyword_only(self, target):
        parameter = signature(target).parameters["description"]
        assert parameter.kind is Parameter.KEYWORD_ONLY
        assert parameter.default is Parameter.empty

    def test_forward_without_backward_raises(self):
        with pytest.raises(ValueError, match="both be set or both be None"):
            OpEntry(
                op="add",
                variant="standard",
                impl="eager",
                description=_TEST_DESCRIPTION,
                forward=_add_forward,
            )

    def test_wrapper_required_when_raw_is_none(self):
        with pytest.raises(ValueError, match="wrapper is required"):
            OpEntry(op="add", variant="standard", impl="eager", description=_TEST_DESCRIPTION)

    def test_description_must_be_non_empty(self):
        with pytest.raises(ValueError, match="description must be a non-empty string"):
            OpEntry(op="add", variant="standard", impl="eager", description=" ", wrapper=lambda: None)

    def test_raw_plus_wrapper_raises(self):
        with pytest.raises(ValueError, match="do not pass wrapper"):
            OpEntry(
                op="add",
                variant="standard",
                impl="eager",
                description=_TEST_DESCRIPTION,
                forward=_add_forward,
                backward=_add_backward,
                wrapper=lambda *args, **kwargs: None,
            )


@pytest.mark.usefixtures("isolated_entries")
class TestRegisterAndResolve:
    def test_register_and_resolve_eager(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        entry = resolve_op("add", "standard", "eager")
        assert entry.forward is _add_forward
        assert entry.backward is _add_backward
        assert entry.description == _TEST_DESCRIPTION
        assert entry.wrapper is not None
        assert "eager" in OP_REGISTRY.list_available("add", "standard")

    def test_unknown_impl_raises_keyerror(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        with pytest.raises(KeyError, match="Unknown op"):
            resolve_op("add", "standard", "missing")

    def test_duplicate_row_raises(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        with pytest.raises(ValueError, match="Duplicate op registration"):
            register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)

    def test_same_impl_can_register_per_device(self):
        OP_REGISTRY.register(
            OpEntry(
                op="add",
                variant="standard",
                impl="fused",
                description="GPU fused add",
                forward=_add_forward,
                backward=_add_backward,
                requirement=_GPU,
            )
        )
        OP_REGISTRY.register(
            OpEntry(
                op="add",
                variant="standard",
                impl="fused",
                description="NPU fused add",
                forward=_add_forward,
                backward=_add_backward,
                requirement=NpuKernelRequirement(),
            )
        )
        OP_REGISTRY.register(
            OpEntry(
                op="add",
                variant="standard",
                impl="fused",
                description="MLU fused add",
                forward=_add_forward,
                backward=_add_backward,
                requirement=MluKernelRequirement(),
            )
        )
        assert OP_REGISTRY.list_registered("add", "standard") == ["fused"]
        assert ("add", "standard", "fused", "cuda") in OP_REGISTRY._entries
        assert ("add", "standard", "fused", "mlu") in OP_REGISTRY._entries
        assert ("add", "standard", "fused", "npu") in OP_REGISTRY._entries
        entries = OP_REGISTRY.list_entries("add", "standard")
        assert [entry.description for entry in entries] == ["GPU fused add", "NPU fused add", "MLU fused add"]
        assert OP_REGISTRY.list_entries("add", "other") == []
        with pytest.raises(ValueError, match="device='cuda'"):
            OP_REGISTRY.register(
                OpEntry(
                    op="add",
                    variant="standard",
                    impl="fused",
                    description="Duplicate GPU fused add",
                    forward=_add_forward,
                    backward=_add_backward,
                    requirement=_GPU,
                )
            )

    def test_resolve_uses_current_device_then_any(self, monkeypatch):
        def cuda_wrapper(x: Tensor) -> Tensor:
            return x + 1

        def mlu_wrapper(x: Tensor) -> Tensor:
            return x + 2

        def any_wrapper(x: Tensor) -> Tensor:
            return x + 3

        register_op(
            "add",
            "standard",
            "fused",
            description="GPU fused add",
            wrapper=cuda_wrapper,
            requirement=_GPU,
        )
        register_op(
            "add",
            "standard",
            "fused",
            description="MLU fused add",
            wrapper=mlu_wrapper,
            requirement=MluKernelRequirement(),
        )
        register_op("add", "standard", "eager", description="Eager add", wrapper=any_wrapper)
        monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "mlu")
        monkeypatch.setattr("veomni.ops.platform.requirement.IS_MLU_AVAILABLE", True)
        assert resolve_op("add", "standard", "fused").wrapper is mlu_wrapper
        assert resolve_op("add", "standard", "eager").wrapper is any_wrapper
        monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "cpu")
        with pytest.raises(RuntimeError, match="not registered for device 'cpu'"):
            resolve_op("add", "standard", "fused")
        assert ("add", "standard", "eager", ANY_DEVICE) in OP_REGISTRY._entries

    def test_unmatched_requirement_is_registered_but_not_resolvable(self, monkeypatch):
        monkeypatch.setattr("veomni.ops.registry.get_device_type", lambda: "cuda")
        register_op(
            "add",
            "standard",
            "cuda_only",
            _add_forward,
            _add_backward,
            description="CUDA-only add",
            requirement=GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=999),)),
        )
        assert "cuda_only" in OP_REGISTRY.list_registered("add", "standard")
        assert "cuda_only" not in OP_REGISTRY.list_available("add", "standard")
        with pytest.raises(RuntimeError, match="requirement is not satisfied"):
            resolve_op("add", "standard", "cuda_only")
        with pytest.raises(RuntimeError, match="requirement is not satisfied"):
            VeomniOp("add", "standard", "cuda_only")

    def test_register_rejects_non_entry(self):
        with pytest.raises(TypeError, match="OpEntry"):
            OP_REGISTRY.register(object())  # type: ignore[arg-type]

    def test_resolve_returns_same_entry(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        entry = OP_REGISTRY.resolve("add", "standard", "eager")
        assert entry is resolve_op("add", "standard", "eager")

    def test_opaque_wrapper_has_no_raw(self):
        def opaque(x: Tensor) -> Tensor:
            return x * 2

        register_op("scale", "standard", "eager", description="Opaque scale", wrapper=opaque)
        entry = resolve_op("scale", "standard", "eager")
        assert entry.forward is None
        assert entry.backward is None
        x = torch.tensor([1.0, 2.0])
        assert torch.equal(entry.wrapper(x), x * 2)
        with pytest.raises(TypeError):
            entry.forward(x)


@pytest.mark.usefixtures("isolated_entries")
class TestGeneratedWrapper:
    def test_wrapper_matches_raw_grads(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        entry = resolve_op("add", "standard", "eager")
        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        scale = 2.5

        out = entry.wrapper(x, y, scale=scale)
        go = torch.randn_like(out)
        out.backward(go)

        raw_out, saved = entry.forward(x.detach(), y.detach(), scale=scale)
        gx, gy = entry.backward(go, saved)
        assert torch.allclose(out, raw_out)
        assert torch.allclose(x.grad, gx)
        assert torch.allclose(y.grad, gy)

    def test_multi_output_wrapper(self):
        register_op("pair", "standard", "eager", _pair_forward, _pair_backward, description=_TEST_DESCRIPTION)
        entry = resolve_op("pair", "standard", "eager")
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        summed, prod = entry.wrapper(x, y)
        assert isinstance(summed, Tensor)
        assert isinstance(prod, Tensor)
        (summed + prod).sum().backward()
        assert x.grad is not None
        assert y.grad is not None

    @pytest.mark.parametrize("pass_as_keyword", (False, True))
    def test_optional_positional_tensor_uses_full_autograd_signature(self, pass_as_keyword):
        register_op(
            "optional_add",
            "standard",
            "eager",
            _optional_forward,
            _optional_backward,
            description=_TEST_DESCRIPTION,
        )
        entry = resolve_op("optional_add", "standard", "eager")
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True) if pass_as_keyword else None

        output = entry.wrapper(x, y=y) if pass_as_keyword else entry.wrapper(x)
        output.sum().backward()

        assert torch.equal(x.grad, torch.ones_like(x))
        if y is not None:
            assert torch.equal(y.grad, torch.ones_like(y))


@pytest.mark.usefixtures("isolated_entries")
class TestVeomniOp:
    def test_call_equals_wrapper(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        handle = VeomniOp("add", "standard", "eager")
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        assert torch.equal(handle(x, y, scale=1.0), resolve_op("add", "standard", "eager").wrapper(x, y, scale=1.0))

    def test_two_impls_stay_local(self):
        def scale2_forward(x: Tensor, y: Tensor, *, scale: float) -> tuple[Tensor, SavedState]:
            return _add_forward(x, y, scale=scale * 2)

        register_op("add", "standard", "eager", _add_forward, _add_backward, description="Eager add")
        register_op("add", "standard", "double", scale2_forward, _add_backward, description="Doubled add")
        eager = VeomniOp("add", "standard", "eager")
        double = VeomniOp("add", "standard", "double")
        x = torch.tensor([1.0])
        y = torch.tensor([1.0])
        assert not torch.equal(eager(x, y, scale=1.0), double(x, y, scale=1.0))
        assert eager is not double

    def test_intern_by_triple(self):
        register_op("add", "standard", "eager", _add_forward, _add_backward, description=_TEST_DESCRIPTION)
        assert VeomniOp("add", "standard") is VeomniOp("add", "standard", "eager")
