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

"""Tests for the ``linear`` operation family (``standard``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp
from veomni.ops.kernels.linear._module import QUANTIZING_IMPLS
from veomni.utils.import_utils import is_package_available


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_linear_registry_entries():
    assert OP_REGISTRY.list_registered("linear", "standard") == ["eager", "qat", "torchao_nvfp4"]
    assert OP_REGISTRY.list_registered("linear", "grouped") == ["eager", "qat"]
    assert OP_REGISTRY.default_impl("linear", "standard", device="cpu") == "eager"
    assert OP_REGISTRY.default_impl("linear", "grouped", device="cpu") == "eager"


def test_linear_eager_matches_f_linear():
    op = VeomniOp("linear", "standard", "eager")
    x = torch.randn(2, 3, 16)
    linear = torch.nn.Linear(16, 8)
    torch.testing.assert_close(op(linear, x), F.linear(x, linear.weight, linear.bias))

    no_bias = torch.nn.Linear(16, 8, bias=False)
    torch.testing.assert_close(op(no_bias, x), F.linear(x, no_bias.weight, None))


def test_linear_eager_backward_matches_f_linear():
    op = VeomniOp("linear", "standard", "eager")
    x = torch.randn(4, 16, requires_grad=True)
    linear = torch.nn.Linear(16, 8)
    w, b = linear.weight, linear.bias
    op(linear, x).sum().backward()

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    F.linear(x_ref, w_ref, b_ref).sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(w.grad, w_ref.grad)
    torch.testing.assert_close(b.grad, b_ref.grad)


def test_linear_eager_handles_empty_input():
    op = VeomniOp("linear", "standard", "eager")
    x = torch.randn(0, 16)
    assert op(torch.nn.Linear(16, 8), x).shape == (0, 8)


def test_linear_qat_rows_require_tilelang_on_sm90():
    """The TileLang rows are declared with the hardware and package they need."""
    entries = {(entry.impl, entry.requires) for entry in OP_REGISTRY.list_entries("linear", "standard")}
    assert ("qat", ("tilelang",)) in entries
    assert ("qat", ("tilelang",)) in {
        (entry.impl, entry.requires) for entry in OP_REGISTRY.list_entries("linear", "grouped")
    }


def test_a_wrapped_projection_falls_back_to_its_own_forward():
    """A wrapped projection keeps its base layer at ``base_layer`` and owns no ``weight``."""
    from torch import nn

    class Wrapped(nn.Module):
        """Stand-in for ``LoraLinear``: no ``weight`` attribute, own forward."""

        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(8, 4, bias=False)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.base_layer(x)

    module = Wrapped()
    x = torch.randn(2, 8)
    op = VeomniOp("linear", "standard", "eager")

    out = op(module, x)

    assert module.calls == 1
    # A real forward, not a reimplementation: the adapter's own math ran.
    torch.testing.assert_close(out, module.base_layer(x))


@pytest.mark.parametrize("quantizing_impl", sorted(QUANTIZING_IMPLS))
def test_a_quantizing_row_refuses_a_wrapped_projection(quantizing_impl):
    """Running the unquantized base layer would leave the config claiming QAT."""
    from torch import nn

    from veomni.ops.kernels.linear._module import resolve_projection

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.base_layer(x)

    with pytest.raises(ValueError, match="cannot run on .Wrapped."):
        resolve_projection(quantizing_impl, Wrapped(), torch.randn(2, 8))


class _GroupedProjection(torch.nn.Linear):
    """The grouped layout ``DeepseekV4GroupedLinear`` owns: one flat weight, a bmm forward."""

    def __init__(self, in_per_group, out_features, n_groups):
        super().__init__(in_per_group, out_features, bias=False)
        self.n_groups = n_groups
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        input_shape = x.shape[:-2]
        w = self.weight.view(self.n_groups, -1, x.shape[-1]).transpose(1, 2)
        grouped = x.reshape(-1, self.n_groups, x.shape[-1]).transpose(0, 1)
        return torch.bmm(grouped, w).transpose(0, 1).reshape(*input_shape, self.n_groups, -1)


def test_the_grouped_row_calls_the_projection_module():
    """The row delegates to the module, so a per-module FSDP2 hook still runs.

    Driving ``bmm`` from ``linear.weight`` directly would skip the module's
    ``forward``; a projection listed in ``basic_modules`` is ``fully_shard``-wrapped
    on its own and unshards its weight in that hook.
    """
    n_groups, in_per_group, out_per_group = 2, 8, 4
    linear = _GroupedProjection(in_per_group, n_groups * out_per_group, n_groups)
    x = torch.randn(3, n_groups, in_per_group)

    out = VeomniOp("linear", "grouped", "eager")(linear, x)

    assert linear.calls == 1
    expected = torch.bmm(
        x.reshape(-1, n_groups, in_per_group).transpose(0, 1),
        linear.weight.view(n_groups, out_per_group, in_per_group).transpose(1, 2),
    ).transpose(0, 1)
    torch.testing.assert_close(out.reshape(-1, n_groups, out_per_group), expected)


def test_the_standard_eager_row_calls_the_projection_module():
    """Same reason as the grouped row: a ``basic_modules`` wrap unshards in ``forward``.

    Reading ``linear.weight`` here would bypass that hook and hand the row a
    sharded ``DTensor``.
    """
    from torch import nn

    class Probe(nn.Linear):
        def __init__(self):
            super().__init__(16, 8)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return super().forward(x)

    linear = Probe()
    x = torch.randn(2, 16)

    out = VeomniOp("linear", "standard", "eager")(linear, x)

    assert linear.calls == 1
    torch.testing.assert_close(out, F.linear(x, linear.weight, linear.bias))


_requires_torchao = pytest.mark.skipif(not is_package_available("torchao"), reason="torchao_nvfp4 row needs torchao")


def _nvfp4_reference(x: torch.Tensor) -> torch.Tensor:
    """torchao's own 16-wide two-level NVFP4 round trip, computed without the row."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

    x_2d = x.detach().reshape(-1, x.shape[-1])
    quantized = NVFP4Tensor.to_nvfp4(
        x_2d,
        block_size=16,
        per_tensor_scale=per_tensor_amax_to_scale(torch.max(torch.abs(x_2d))),
        is_swizzled_scales=False,
        use_triton_kernel=False,
    )
    return quantized.dequantize(x.dtype).reshape_as(x)


def test_linear_torchao_nvfp4_row_is_declared_with_torchao():
    """The row needs no specific hardware, only the optional package."""
    entries = {(entry.impl, entry.requires) for entry in OP_REGISTRY.list_entries("linear", "standard")}
    assert ("torchao_nvfp4", ("torchao",)) in entries
    assert "torchao_nvfp4" in QUANTIZING_IMPLS
    # ``standard`` only: a model that also binds ``grouped`` cannot select it.
    assert "torchao_nvfp4" not in OP_REGISTRY.list_registered("linear", "grouped")


@_requires_torchao
def test_linear_torchao_nvfp4_forward_and_backward_round_trip():
    """The row equals F.linear over torchao's NVFP4 round trip, with straight-through gradients."""
    from veomni.ops.kernels.linear.standard import torchao_nvfp4 as impl
    from veomni.ops.qat.torchao_nvfp4 import nvfp4_fake_quant

    assert impl.NVFP4_BLOCK_SIZE == 16
    op = VeomniOp("linear", "standard", "torchao_nvfp4")
    torch.manual_seed(0)
    # 48 is a whole number of 16-wide groups: NVFP4 needs nothing wider.
    linear = torch.nn.Linear(48, 32).to(torch.bfloat16)
    x = torch.randn(2, 5, 48, dtype=torch.bfloat16, requires_grad=True)

    ref_x = _nvfp4_reference(x)
    torch.testing.assert_close(nvfp4_fake_quant(x.detach()), ref_x, rtol=0, atol=0)

    out = op(linear, x)
    expected = F.linear(ref_x, _nvfp4_reference(linear.weight), linear.bias.detach())
    torch.testing.assert_close(out, expected)
    assert not torch.equal(out, linear(x))

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()
    torch.testing.assert_close(linear.bias.grad, torch.full_like(linear.bias.grad, 10.0))


@_requires_torchao
def test_linear_torchao_nvfp4_validates_its_operands():
    """16-wide groups only, BF16/FP32 only, and no partial group."""
    from veomni.ops.qat.torchao_nvfp4 import nvfp4_fake_quant

    op = VeomniOp("linear", "standard", "torchao_nvfp4")
    with pytest.raises(ValueError, match="divisible by block_size 16"):
        op(torch.nn.Linear(24, 16, bias=False).to(torch.bfloat16), torch.randn(2, 24, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="requires block_size=16"):
        nvfp4_fake_quant(torch.randn(2, 32, dtype=torch.bfloat16), block_size=32)
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        op(torch.nn.Linear(16, 16, bias=False).to(torch.float16), torch.randn(2, 16, dtype=torch.float16))
    # An all-zero operand dequantizes to zeros instead of 0/0 NaNs.
    zeros = torch.zeros(3, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(nvfp4_fake_quant(zeros), zeros, rtol=0, atol=0)


@_requires_torchao
def test_linear_torchao_nvfp4_handles_empty_input():
    """A zero-token micro-batch keeps its shape and yields zero weight gradients."""
    op = VeomniOp("linear", "standard", "torchao_nvfp4")
    linear = torch.nn.Linear(16, 8).to(torch.bfloat16)
    x = torch.randn(0, 16, dtype=torch.bfloat16, requires_grad=True)

    out = op(linear, x)
    assert out.shape == (0, 8)
    out.sum().backward()
    torch.testing.assert_close(linear.weight.grad, torch.zeros_like(linear.weight))
