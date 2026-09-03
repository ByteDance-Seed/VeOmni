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

"""CPU tests for KernelEntry / VeomniKernel."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from veomni.kernels import KERNEL_REGISTRY, VeomniKernel, register_kernel, resolve_kernel
from veomni.kernels.registry import KernelEntry, SavedState
from veomni.kernels.requirement import ANY_DEVICE, CudaKernelRequirement, MluKernelRequirement, NpuKernelRequirement


@pytest.fixture
def isolated_entries():
    saved_entries = dict(KERNEL_REGISTRY._entries)
    saved_intern = dict(VeomniKernel._intern)
    KERNEL_REGISTRY._entries.clear()
    VeomniKernel._intern.clear()
    yield
    KERNEL_REGISTRY._entries.clear()
    KERNEL_REGISTRY._entries.update(saved_entries)
    VeomniKernel._intern.clear()
    VeomniKernel._intern.update(saved_intern)


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


@pytest.mark.usefixtures("isolated_entries")
class TestKernelEntryValidation:
    def test_forward_without_backward_raises(self):
        with pytest.raises(ValueError, match="both be set or both be None"):
            KernelEntry(kernel="add", variant="standard", impl="eager", forward=_add_forward)

    def test_wrapper_required_when_raw_is_none(self):
        with pytest.raises(ValueError, match="wrapper is required"):
            KernelEntry(kernel="add", variant="standard", impl="eager")

    def test_raw_plus_wrapper_raises(self):
        with pytest.raises(ValueError, match="do not pass wrapper"):
            KernelEntry(
                kernel="add",
                variant="standard",
                impl="eager",
                forward=_add_forward,
                backward=_add_backward,
                wrapper=lambda *args, **kwargs: None,
            )


@pytest.mark.usefixtures("isolated_entries")
class TestRegisterAndResolve:
    def test_register_and_resolve_eager(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        entry = resolve_kernel("add", "standard", "eager")
        assert entry.forward is _add_forward
        assert entry.backward is _add_backward
        assert entry.wrapper is not None
        assert "eager" in KERNEL_REGISTRY.list_available("add", "standard")

    def test_unknown_impl_raises_keyerror(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        with pytest.raises(KeyError, match="Unknown kernel"):
            resolve_kernel("add", "standard", "missing")

    def test_duplicate_row_raises(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        with pytest.raises(ValueError, match="Duplicate kernel registration"):
            register_kernel("add", "standard", "eager", _add_forward, _add_backward)

    def test_same_impl_can_register_per_device(self):
        KERNEL_REGISTRY.register(
            KernelEntry(
                kernel="add",
                variant="standard",
                impl="fused",
                forward=_add_forward,
                backward=_add_backward,
                requirement=CudaKernelRequirement(),
            )
        )
        KERNEL_REGISTRY.register(
            KernelEntry(
                kernel="add",
                variant="standard",
                impl="fused",
                forward=_add_forward,
                backward=_add_backward,
                requirement=NpuKernelRequirement(),
            )
        )
        KERNEL_REGISTRY.register(
            KernelEntry(
                kernel="add",
                variant="standard",
                impl="fused",
                forward=_add_forward,
                backward=_add_backward,
                requirement=MluKernelRequirement(),
            )
        )
        assert KERNEL_REGISTRY.list_registered("add", "standard") == ["fused"]
        assert ("add", "standard", "fused", "cuda") in KERNEL_REGISTRY._entries
        assert ("add", "standard", "fused", "mlu") in KERNEL_REGISTRY._entries
        assert ("add", "standard", "fused", "npu") in KERNEL_REGISTRY._entries
        with pytest.raises(ValueError, match="device='cuda'"):
            KERNEL_REGISTRY.register(
                KernelEntry(
                    kernel="add",
                    variant="standard",
                    impl="fused",
                    forward=_add_forward,
                    backward=_add_backward,
                    requirement=CudaKernelRequirement(),
                )
            )

    def test_resolve_uses_current_device_then_any(self, monkeypatch):
        def cuda_wrapper(x: Tensor) -> Tensor:
            return x + 1

        def mlu_wrapper(x: Tensor) -> Tensor:
            return x + 2

        def any_wrapper(x: Tensor) -> Tensor:
            return x + 3

        register_kernel("add", "standard", "fused", wrapper=cuda_wrapper, requirement=CudaKernelRequirement())
        register_kernel("add", "standard", "fused", wrapper=mlu_wrapper, requirement=MluKernelRequirement())
        register_kernel("add", "standard", "eager", wrapper=any_wrapper)
        monkeypatch.setattr("veomni.kernels.registry.get_device_type", lambda: "mlu")
        monkeypatch.setattr("veomni.kernels.requirement.IS_MLU_AVAILABLE", True)
        assert resolve_kernel("add", "standard", "fused").wrapper is mlu_wrapper
        assert resolve_kernel("add", "standard", "eager").wrapper is any_wrapper
        monkeypatch.setattr("veomni.kernels.registry.get_device_type", lambda: "cpu")
        with pytest.raises(RuntimeError, match="not registered for device 'cpu'"):
            resolve_kernel("add", "standard", "fused")
        assert ("add", "standard", "eager", ANY_DEVICE) in KERNEL_REGISTRY._entries

    def test_unmatched_requirement_is_registered_but_not_resolvable(self):
        register_kernel(
            "add",
            "standard",
            "cuda_only",
            _add_forward,
            _add_backward,
            requirement=CudaKernelRequirement(min_cc=999),
        )
        assert "cuda_only" in KERNEL_REGISTRY.list_registered("add", "standard")
        assert "cuda_only" not in KERNEL_REGISTRY.list_available("add", "standard")
        with pytest.raises(RuntimeError, match="requirement is not satisfied"):
            resolve_kernel("add", "standard", "cuda_only")
        with pytest.raises(RuntimeError, match="requirement is not satisfied"):
            VeomniKernel("add", "standard", "cuda_only")

    def test_register_rejects_non_entry(self):
        with pytest.raises(TypeError, match="KernelEntry"):
            KERNEL_REGISTRY.register(object())  # type: ignore[arg-type]

    def test_resolve_returns_same_entry(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        entry = KERNEL_REGISTRY.resolve("add", "standard", "eager")
        assert entry is resolve_kernel("add", "standard", "eager")

    def test_opaque_wrapper_has_no_raw(self):
        def opaque(x: Tensor) -> Tensor:
            return x * 2

        register_kernel("scale", "standard", "eager", wrapper=opaque)
        entry = resolve_kernel("scale", "standard", "eager")
        assert entry.forward is None
        assert entry.backward is None
        x = torch.tensor([1.0, 2.0])
        assert torch.equal(entry.wrapper(x), x * 2)
        with pytest.raises(TypeError):
            entry.forward(x)


@pytest.mark.usefixtures("isolated_entries")
class TestGeneratedWrapper:
    def test_wrapper_matches_raw_grads(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        entry = resolve_kernel("add", "standard", "eager")
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
        register_kernel("pair", "standard", "eager", _pair_forward, _pair_backward)
        entry = resolve_kernel("pair", "standard", "eager")
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        summed, prod = entry.wrapper(x, y)
        assert isinstance(summed, Tensor)
        assert isinstance(prod, Tensor)
        (summed + prod).sum().backward()
        assert x.grad is not None
        assert y.grad is not None


@pytest.mark.usefixtures("isolated_entries")
class TestVeomniKernel:
    def test_call_equals_wrapper(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        handle = VeomniKernel("add", "standard", "eager")
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        assert torch.equal(
            handle(x, y, scale=1.0), resolve_kernel("add", "standard", "eager").wrapper(x, y, scale=1.0)
        )

    def test_two_impls_stay_local(self):
        def scale2_forward(x: Tensor, y: Tensor, *, scale: float) -> tuple[Tensor, SavedState]:
            return _add_forward(x, y, scale=scale * 2)

        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        register_kernel("add", "standard", "double", scale2_forward, _add_backward)
        eager = VeomniKernel("add", "standard", "eager")
        double = VeomniKernel("add", "standard", "double")
        x = torch.tensor([1.0])
        y = torch.tensor([1.0])
        assert not torch.equal(eager(x, y, scale=1.0), double(x, y, scale=1.0))
        assert eager is not double

    def test_intern_by_triple(self):
        register_kernel("add", "standard", "eager", _add_forward, _add_backward)
        assert VeomniKernel("add", "standard") is VeomniKernel("add", "standard", "eager")
