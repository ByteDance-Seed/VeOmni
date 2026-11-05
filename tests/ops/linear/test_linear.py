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
from veomni.utils.import_utils import is_package_available


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

_requires_torchao = pytest.mark.skipif(
    not is_package_available("torchao"),
    reason="torchao_mxfp4 row requires the optional 'torchao' package",
)


def _empty_bias(weight: torch.Tensor) -> torch.Tensor:
    return weight.new_empty(0)


def test_linear_registry_entries():
    assert OP_REGISTRY.list_registered("linear", "standard") == ["eager", "torchao_mxfp4"]
    assert OP_REGISTRY.default_impl("linear", "standard", device="cpu") == "eager"


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


def test_linear_explicit_unavailable_impl_raises():
    """A row whose package is missing must fail loudly rather than fall back to eager."""
    if is_package_available("torchao"):
        pytest.skip("torchao is installed, so the row resolves")
    with pytest.raises(RuntimeError, match="requires unavailable package"):
        VeomniOp("linear", "standard", "torchao_mxfp4")


@_requires_torchao
def test_linear_torchao_mxfp4_forward_matches_manual_round_trip():
    from veomni.ops.qat.torchao_mxfp4 import mxfp4_fake_quant

    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    w = torch.randn(32, 64, dtype=torch.bfloat16)
    b = torch.randn(32, dtype=torch.bfloat16)

    expected = F.linear(mxfp4_fake_quant(x), mxfp4_fake_quant(w), b)
    torch.testing.assert_close(op(x, w, b), expected)


@_requires_torchao
def test_linear_torchao_mxfp4_actually_rounds():
    """The row must change the numbers; passing through unquantized would be a silent no-op."""
    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    w = torch.randn(32, 64, dtype=torch.bfloat16)
    assert not torch.equal(op(x, w, _empty_bias(w)), F.linear(x, w, None))


@_requires_torchao
def test_linear_torchao_mxfp4_backward_is_straight_through():
    """Gradients use the quantized operands, matching torchao's MX QAT backward."""
    from veomni.ops.qat.torchao_mxfp4 import mxfp4_fake_quant

    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(32, 64, dtype=torch.bfloat16, requires_grad=True)
    op(x, w, _empty_bias(w)).sum().backward()

    grad_out = torch.ones(4, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(
        x.grad.float(), grad_out.float() @ mxfp4_fake_quant(w).detach().float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        w.grad.float(), grad_out.t().float() @ mxfp4_fake_quant(x).detach().float(), rtol=1e-2, atol=1e-2
    )


@_requires_torchao
def test_linear_torchao_mxfp4_bias_gradient():
    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    w = torch.randn(32, 64, dtype=torch.bfloat16)
    b = torch.randn(32, dtype=torch.bfloat16, requires_grad=True)
    op(x, w, b).sum().backward()
    torch.testing.assert_close(b.grad, torch.full_like(b.grad, 10.0))


@_requires_torchao
def test_linear_torchao_mxfp4_rejects_partial_group():
    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    x = torch.randn(2, 48, dtype=torch.bfloat16)
    w = torch.randn(32, 48, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by block_size"):
        op(x, w, _empty_bias(w))


@_requires_torchao
def test_linear_torchao_mxfp4_handles_empty_input():
    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    x = torch.randn(0, 64, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(32, 64, dtype=torch.bfloat16, requires_grad=True)
    out = op(x, w, _empty_bias(w))
    assert out.shape == (0, 32)

    # A zero-token micro-batch must still reach the weights with zero gradients.
    out.sum().backward()
    torch.testing.assert_close(w.grad, torch.zeros_like(w.grad))


@_requires_torchao
def test_linear_torchao_mxfp4_accepts_non_contiguous_input():
    """A sliced activation arrives non-contiguous and torchao's to_mx asserts contiguity."""
    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    w = torch.randn(32, 64, dtype=torch.bfloat16)
    sliced = torch.randn(3, 8, 64, dtype=torch.bfloat16)[:, :5]
    assert not sliced.is_contiguous()
    assert op(sliced, w, _empty_bias(w)).shape == (3, 5, 32)


@_requires_torchao
def test_linear_torchao_mxfp4_rejects_float16():
    op = VeomniOp("linear", "standard", "torchao_mxfp4")
    x = torch.randn(2, 64, dtype=torch.float16)
    w = torch.randn(32, 64, dtype=torch.float16)
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        op(x, w, _empty_bias(w))


@_requires_torchao
def test_linear_torchao_mxfp4_default_block_size_is_the_ocp_width():
    """The default must be the 32-wide deployment group; a different width silently retrains a different recipe."""
    from veomni.ops.kernels.linear.standard import torchao_mxfp4 as impl

    assert impl.MXFP4_BLOCK_SIZE == 32
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    w = torch.randn(8, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(
        impl.wrapper(x, w, _empty_bias(w)),
        impl.wrapper(x, w, _empty_bias(w), block_size=32),
    )


@_requires_torchao
@pytest.mark.parametrize("block_size", [16, 64])
def test_linear_torchao_mxfp4_block_size_changes_the_rounding(block_size: int):
    """``block_size`` reaches the quantizer rather than being accepted and dropped."""
    from veomni.ops.kernels.linear.standard import torchao_mxfp4 as impl

    torch.manual_seed(0)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    w = torch.randn(8, 128, dtype=torch.bfloat16)
    default = impl.wrapper(x, w, _empty_bias(w))
    assert not torch.equal(default, impl.wrapper(x, w, _empty_bias(w), block_size=block_size))


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


@_requires_torchao
def test_linear_torchao_mxfp4_accepts_replicated_dtensor_parameter(single_rank_dist):
    """FSDP2 hands the projection a DTensor in some placements; quantizing the local shard must work.

    Returning a plain tensor for the weight while leaving the bias a DTensor makes
    ``addmm`` reject the call with "got mixed torch.Tensor and DTensor", so both
    operands take the same path.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    weight = distribute_tensor(torch.randn(32, 64, dtype=torch.bfloat16), mesh, [Replicate()])
    bias = distribute_tensor(torch.randn(32, dtype=torch.bfloat16), mesh, [Replicate()])
    x = torch.randn(2, 64, dtype=torch.bfloat16)

    out = VeomniOp("linear", "standard", "torchao_mxfp4")(x, weight, bias)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 32)


@_requires_torchao
def test_linear_torchao_mxfp4_accepts_replicated_dtensor_with_empty_input(single_rank_dist):
    """The zero-token path must not be the one branch that skips the placement normalization."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    weight = distribute_tensor(torch.randn(32, 64, dtype=torch.bfloat16), mesh, [Replicate()])
    bias = distribute_tensor(torch.randn(32, dtype=torch.bfloat16), mesh, [Replicate()])
    x = torch.randn(0, 64, dtype=torch.bfloat16)

    assert VeomniOp("linear", "standard", "torchao_mxfp4")(x, weight, bias).shape == (0, 32)


@_requires_torchao
def test_linear_torchao_mxfp4_rejects_sharded_dtensor_parameter(single_rank_dist):
    """A sharded placement would split a 32-wide group across shards."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    weight = distribute_tensor(torch.randn(32, 64, dtype=torch.bfloat16), mesh, [Shard(1)])
    x = torch.randn(2, 64, dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="replicated DTensor"):
        VeomniOp("linear", "standard", "torchao_mxfp4")(x, weight, weight.new_empty(0))
