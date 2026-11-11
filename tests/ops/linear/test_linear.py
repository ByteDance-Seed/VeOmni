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


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_linear_registry_entries():
    assert OP_REGISTRY.list_registered("linear", "standard") == ["eager", "qat"]
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


@pytest.fixture
def single_rank_dist(monkeypatch):
    """Provide a one-rank gloo group and destroy it, so later tests see no leaked state."""
    import torch.distributed as dist

    from tests.tools.launch_utils import find_free_port

    created = False
    if not dist.is_initialized():
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(find_free_port()))
        dist.init_process_group("gloo", rank=0, world_size=1)
        created = True
    try:
        yield
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


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
