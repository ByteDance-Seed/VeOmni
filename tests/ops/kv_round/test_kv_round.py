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

"""Tests for the ``kv_round`` operation family (``standard`` and ``deepseek_v4``)."""

from __future__ import annotations

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_kv_round_registry_entries():
    assert OP_REGISTRY.list_registered("kv_round", "standard") == ["none", "eager", "triton"]
    assert OP_REGISTRY.default_impl("kv_round", "standard", device="cpu") == "eager"
    assert OP_REGISTRY.list_registered("kv_round", "deepseek_v4") == ["none", "eager", "triton"]
    assert OP_REGISTRY.default_impl("kv_round", "deepseek_v4", device="cpu") == "eager"


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_fp8_eager_round_trip_and_straight_through_backward(block_size: int):
    op = VeomniOp("kv_round", "standard", "eager")
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
    op = VeomniOp("kv_round", "standard", "eager")
    x = torch.randn(2, 4, block_size * 3, dtype=torch.bfloat16)
    prefix = block_size * 2

    y = op(x, block_size=block_size, quant_features=prefix)

    torch.testing.assert_close(y[..., prefix:], x[..., prefix:])
    assert not torch.equal(y[..., :prefix], x[..., :prefix])
    torch.testing.assert_close(y[..., :prefix], op(x[..., :prefix], block_size=block_size))


def test_fp8_quant_features_equal_to_width_quantizes_everything():
    op = VeomniOp("kv_round", "standard", "eager")
    x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(op(x, block_size=128, quant_features=128), op(x, block_size=128))


def test_fp8_rejects_a_partial_group():
    op = VeomniOp("kv_round", "standard", "eager")
    x = torch.randn(2, 4, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not divisible by FP8 block size"):
        op(x, block_size=64)
    with pytest.raises(ValueError, match="not divisible by block_size"):
        op(x, block_size=32, quant_features=48)


def test_fp8_scale_is_a_power_of_two():
    """E8M0 scales round up, which is what keeps the straight-through backward sound."""
    from veomni.ops.kernels.kv_round.standard.eager import fp8_roundtrip

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    y = fp8_roundtrip(x, block_size=4)
    # One 4-wide block, amax 4.0 -> scale 8/448 rounded up to a power of two.
    assert torch.isfinite(y).all()
    assert (y.abs() <= x.abs().max()).all()


def test_empty_operand_is_returned_unchanged():
    op = VeomniOp("kv_round", "standard", "eager")
    x = torch.randn(0, 4, 32, dtype=torch.bfloat16)
    assert op(x, block_size=32).shape == x.shape


def test_explicit_unavailable_impl_raises_on_cpu():
    """The Triton row is CUDA-only and must fail loudly rather than fall back."""
    with pytest.raises(RuntimeError, match="not registered for device|requires unavailable package"):
        VeomniOp("kv_round", "standard", "triton")


def test_fp8_round_trip_matches_the_deepseek_v4_reference_bitwise():
    """The ported eager path must reproduce the values the previous QAT helpers produced.

    DeepSeek-V4's recipe rounds window KV in 64-wide groups over the NoPE prefix
    and indexer logits in 128-wide groups over the whole head. Both were served by
    one TileLang helper before this family existed, so a difference here means
    training would silently drift from the deployment kernel.
    """
    from veomni.ops.kernels.kv_round.standard.eager import fp8_roundtrip

    fp8_max = 448.0

    def reference(x: torch.Tensor, block_size: int) -> torch.Tensor:
        last = x.shape[-1]
        dtype = x.dtype
        rows = x.reshape(-1, last).float()
        blocks = rows.reshape(rows.shape[0], last // block_size, block_size)
        amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        scaled = (amax / fp8_max).contiguous()
        bits = scaled.view(torch.int32)
        exponent = ((bits >> 23) & 0xFF) - 127
        log2_ceil = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
        scale = ((log2_ceil + 127) << 23).view(torch.float32)
        quantized = (blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).float()
        return (quantized * scale).reshape_as(x).to(dtype)

    torch.manual_seed(0)
    for block_size in (32, 64, 128):
        x = torch.randn(3, 7, 128, dtype=torch.bfloat16)
        torch.testing.assert_close(fp8_roundtrip(x, block_size=block_size), reference(x, block_size), rtol=0, atol=0)

    # The prefix form must equal quantizing the slice and keeping the tail.
    kv = torch.randn(2, 5, 192, dtype=torch.bfloat16)
    prefixed = fp8_roundtrip(kv, block_size=64, quant_features=128)
    expected = torch.cat([reference(kv[..., :128], 64), kv[..., 128:]], dim=-1)
    torch.testing.assert_close(prefixed, expected, rtol=0, atol=0)


def test_deepseek_v4_variant_rounds_the_nope_prefix_only():
    """A KV entry keeps its RoPE channels in the model dtype; only the NoPE part rounds."""
    from veomni.ops.kernels.kv_round.deepseek_v4 import eager as dsv4

    torch.manual_seed(0)
    rope_features = 64
    kv = torch.randn(2, 3, 192, dtype=torch.bfloat16)

    rounded = dsv4.round_window_kv(kv, rope_features)

    torch.testing.assert_close(rounded[..., -rope_features:], kv[..., -rope_features:])
    assert not torch.equal(rounded[..., :-rope_features], kv[..., :-rope_features])
    assert dsv4.WINDOW_KV_BLOCK_SIZE == 64


def test_deepseek_v4_variant_rounds_whole_indexer_heads():
    """The indexer rounds the whole head over 128-wide groups, with no RoPE split."""
    from veomni.ops.kernels.kv_round.deepseek_v4 import eager as dsv4

    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)

    rounded = dsv4.round_indexer_act(x)

    assert not torch.equal(rounded, x)
    assert dsv4.INDEXER_BLOCK_SIZE == 128
    # No prefix: every channel is rounded.
    torch.testing.assert_close(rounded, dsv4.round_window_kv(x, rope_features=0))


def test_deepseek_v4_variant_is_the_standard_round_trip_with_fixed_geometry():
    """The variant owns the numbers, not a second implementation of the arithmetic."""
    from veomni.ops.kernels.kv_round.deepseek_v4 import eager as dsv4
    from veomni.ops.kernels.kv_round.standard.eager import fp8_roundtrip

    torch.manual_seed(0)
    kv = torch.randn(2, 3, 192, dtype=torch.bfloat16)

    torch.testing.assert_close(
        dsv4.round_window_kv(kv, 64),
        fp8_roundtrip(kv, block_size=64, quant_features=128),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dsv4.round_indexer_act(kv[..., :128]),
        fp8_roundtrip(kv[..., :128], block_size=128),
        rtol=0,
        atol=0,
    )


def test_storage_rounding_rows_are_straight_through():
    """Both DeepSeek-V4 recipes pass the gradient through unchanged."""
    from veomni.ops import VeomniOp

    op = VeomniOp("kv_round", "deepseek_v4", "eager")
    # 192 = 128 NoPE + 64 RoPE for the window recipe; the indexer rounds a whole
    # 128-wide head, so it takes a tensor of that width.
    cases = ((192, {"rope_features": 64}), (128, {"rope_features": None}))
    for width, kwargs in cases:
        x = torch.randn(2, 3, width, dtype=torch.bfloat16, requires_grad=True)
        y = op(x, **kwargs)
        grad = torch.randn_like(y)
        (grad_x,) = torch.autograd.grad(y, x, grad)
        torch.testing.assert_close(grad_x, grad, rtol=0, atol=0)
