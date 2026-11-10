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
from veomni.models.utils.linear_utils import QUANTIZING_LINEAR_IMPLS
from veomni.ops import OP_REGISTRY, VeomniOp


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _empty_bias(weight: torch.Tensor) -> torch.Tensor:
    """The empty-tensor sentinel the family uses for an unused bias."""
    return weight.new_empty(0)


def test_linear_registry_entries():
    assert OP_REGISTRY.list_registered("linear", "standard") == ["eager", "qat"]
    assert OP_REGISTRY.list_registered("linear", "grouped") == ["eager", "qat"]
    assert OP_REGISTRY.default_impl("linear", "standard", device="cpu") == "eager"
    assert OP_REGISTRY.default_impl("linear", "grouped", device="cpu") == "eager"


def test_linear_eager_matches_f_linear():
    op = VeomniOp("linear", "standard", "eager")
    x = torch.randn(2, 3, 16)
    w = torch.randn(8, 16)
    b = torch.randn(8)
    torch.testing.assert_close(op(x, w, b), F.linear(x, w, b))
    torch.testing.assert_close(op(x, w, _empty_bias(w)), F.linear(x, w, None))


def test_linear_eager_backward_matches_f_linear():
    op = VeomniOp("linear", "standard", "eager")
    x = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(8, 16, requires_grad=True)
    b = torch.randn(8, requires_grad=True)
    op(x, w, b).sum().backward()

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
    w = torch.randn(8, 16)
    assert op(x, w, _empty_bias(w)).shape == (0, 8)


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


def test_project_through_op_falls_back_to_a_wrapped_module():
    """A wrapped projection keeps its base layer at ``base_layer`` and owns no ``weight``."""
    from torch import nn

    from veomni.models.utils.linear_utils import project_through_op

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
    handle = VeomniOp("linear", "standard", "eager")

    out = project_through_op(handle, module, x)

    assert module.calls == 1
    torch.testing.assert_close(out, module.base_layer(x))
    # The fallback is a real forward, not a reimplementation.
    torch.testing.assert_close(out, F.linear(x, module.base_layer.weight, None))


@pytest.mark.parametrize("quantizing_impl", sorted(QUANTIZING_LINEAR_IMPLS))
def test_project_through_op_refuses_a_quantizing_row_on_a_wrapped_module(quantizing_impl):
    """Running the unquantized base layer would leave the config claiming QAT."""
    from torch import nn

    from veomni.models.utils.linear_utils import project_through_op

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.base_layer(x)

    class FakeOp:
        impl = quantizing_impl

    with pytest.raises(ValueError, match="cannot run on a wrapped projection"):
        project_through_op(FakeOp(), Wrapped(), torch.randn(2, 8))
