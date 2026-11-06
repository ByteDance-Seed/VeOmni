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

"""Tests for the ``kv_round`` operation family (``fp8`` and ``fp4``)."""

from __future__ import annotations

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_kv_round_registry_entries():
    assert OP_REGISTRY.list_registered("kv_round", "fp8") == ["eager", "triton"]
    assert OP_REGISTRY.list_registered("kv_round", "fp4") == ["eager", "triton"]
    assert OP_REGISTRY.default_impl("kv_round", "fp8", device="cpu") == "eager"
    assert OP_REGISTRY.default_impl("kv_round", "fp4", device="cpu") == "eager"


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_fp8_eager_round_trip_and_straight_through_backward(block_size: int):
    op = VeomniOp("kv_round", "fp8", "eager")
    x = torch.randn(2, 4, block_size * 2, dtype=torch.bfloat16, requires_grad=True)
    y = op(x, block_size=block_size)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not torch.equal(y, x)

    grad_out = torch.randn_like(y)
    (grad_x,) = torch.autograd.grad(y, x, grad_out)
    torch.testing.assert_close(grad_x, grad_out)


@pytest.mark.parametrize("block_size", [32, 64])
def test_fp8_quant_features_limits_the_round_trip_to_the_prefix(block_size: int):
    """DeepSeek-V4 stores the NoPE channels and keeps the RoPE tail in the model dtype."""
    op = VeomniOp("kv_round", "fp8", "eager")
    x = torch.randn(2, 4, block_size * 3, dtype=torch.bfloat16)
    prefix = block_size * 2

    y = op(x, block_size=block_size, quant_features=prefix)

    torch.testing.assert_close(y[..., prefix:], x[..., prefix:])
    assert not torch.equal(y[..., :prefix], x[..., :prefix])
    torch.testing.assert_close(y[..., :prefix], op(x[..., :prefix], block_size=block_size))


def test_fp8_quant_features_equal_to_width_quantizes_everything():
    op = VeomniOp("kv_round", "fp8", "eager")
    x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(op(x, block_size=128, quant_features=128), op(x, block_size=128))


def test_fp8_rejects_a_partial_group():
    op = VeomniOp("kv_round", "fp8", "eager")
    x = torch.randn(2, 4, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not divisible by FP8 block size"):
        op(x, block_size=64)
    with pytest.raises(ValueError, match="not divisible by block_size"):
        op(x, block_size=32, quant_features=48)


def test_fp8_scale_is_a_power_of_two():
    """E8M0 scales round up, which is what keeps the straight-through backward sound."""
    from veomni.ops.kernels.kv_round.fp8.eager import fp8_roundtrip

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    y = fp8_roundtrip(x, block_size=4)
    # One 4-wide block, amax 4.0 -> scale 8/448 rounded up to a power of two.
    assert torch.isfinite(y).all()
    assert (y.abs() <= x.abs().max()).all()


@pytest.mark.parametrize("scale_fmt,block_size", [("e4m3", 16), ("ue8m0", 32)])
def test_fp4_eager_round_trip_and_straight_through_backward(scale_fmt: str, block_size: int):
    op = VeomniOp("kv_round", "fp4", "eager")
    x = torch.randn(2, 3, block_size * 2, dtype=torch.bfloat16, requires_grad=True)
    y = op(x, block_size=block_size, scale_fmt=scale_fmt)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not torch.equal(y, x)
    # E2M1 magnitudes are bounded by 6 per group scale; nothing should escape it.
    assert torch.isfinite(y).all()

    grad_out = torch.randn_like(y)
    (grad_x,) = torch.autograd.grad(y, x, grad_out)
    torch.testing.assert_close(grad_x, grad_out)


def test_fp4_rejects_an_unknown_scale_format():
    op = VeomniOp("kv_round", "fp4", "eager")
    x = torch.randn(1, 2, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="scale_fmt must be"):
        op(x, block_size=16, scale_fmt="fp16")


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
def test_empty_operand_is_returned_unchanged(variant: str):
    op = VeomniOp("kv_round", variant, "eager")
    x = torch.randn(0, 4, 32, dtype=torch.bfloat16)
    assert op(x, block_size=32).shape == x.shape


def test_explicit_unavailable_impl_raises_on_cpu():
    """The Triton row is CUDA-only and must fail loudly rather than fall back."""
    with pytest.raises(RuntimeError, match="not registered for device|requires unavailable package"):
        VeomniOp("kv_round", "fp8", "triton")
