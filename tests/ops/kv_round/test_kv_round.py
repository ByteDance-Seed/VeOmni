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

"""Tests for the ``kv_round`` operation family (``standard``, ``deepseek_v4`` and ``fp4``)."""

from __future__ import annotations

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp
from veomni.utils.device import IS_CUDA_AVAILABLE


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


# ---------------------------------------------------------------------------
# The Triton row
#
# Everything above runs on CPU and exercises ``eager``; ``triton`` is only ever
# asserted to be *unavailable* there. The kernel itself -- the ``float8e4nv``
# cast, the block scan, the prefix split, the straight-through backward -- had no
# coverage at all until these tests. They carry their own launch plan because the
# module-level one pins the whole file to CPU, and ``get_closest_marker`` takes
# the function-level marker over it.
# ---------------------------------------------------------------------------

_TRITON_DEVICE = pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))),
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.launch_plan(provider=LaunchSpec()),
                pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="the kv_round Triton row needs CUDA"),
            ],
        ),
    ],
)


def _triton_op(variant: str, device: str):
    """Return the Triton row, skipping when the registry says it cannot run here.

    The availability check goes through the registry rather than reading a compute
    capability here: the row already declares ``NVIDIA_SM89_PLUS`` and its Triton
    requirement, and duplicating that as a hand-written version test drifts from it
    -- an SM86 host satisfies neither, so it must skip rather than fail.
    """
    if device == "cpu":
        pytest.skip("the kv_round Triton row is CUDA-only")
    pytest.importorskip("triton")
    if "triton" not in OP_REGISTRY.list_available("kv_round", variant):
        pytest.skip("the registry reports no usable kv_round Triton row on this device")
    return VeomniOp("kv_round", variant, "triton")


@_TRITON_DEVICE
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_triton_round_trip_matches_eager(device: str, block_size: int):
    """The kernel must reproduce the eager round trip exactly, not approximately.

    Both rows implement the same recipe -- group amax, round the scale up to a
    power of two, clamp, cast -- so a difference means the deployment numerics
    drifted from what training sees.
    """
    eager = VeomniOp("kv_round", "standard", "eager")
    triton = _triton_op("standard", device)

    torch.manual_seed(0)
    x = torch.randn(3, 5, block_size * 2, dtype=torch.bfloat16, device=device)

    torch.testing.assert_close(triton(x, block_size=block_size), eager(x, block_size=block_size), rtol=0, atol=0)


@_TRITON_DEVICE
@pytest.mark.parametrize("block_size", [32, 64])
def test_triton_prefix_split_matches_eager(device: str, block_size: int):
    """``quant_features`` stops the round trip partway; the tail must stay untouched."""
    eager = VeomniOp("kv_round", "standard", "eager")
    triton = _triton_op("standard", device)

    torch.manual_seed(1)
    x = torch.randn(2, 4, block_size * 3, dtype=torch.bfloat16, device=device)
    prefix = block_size * 2

    actual = triton(x, block_size=block_size, quant_features=prefix)

    torch.testing.assert_close(actual, eager(x, block_size=block_size, quant_features=prefix), rtol=0, atol=0)
    torch.testing.assert_close(actual[..., prefix:], x[..., prefix:], rtol=0, atol=0)


@_TRITON_DEVICE
def test_triton_deepseek_v4_variants_match_eager(device: str):
    """Both model geometries go through the same kernel; both must agree with eager."""
    eager = VeomniOp("kv_round", "deepseek_v4", "eager")
    triton = _triton_op("deepseek_v4", device)

    torch.manual_seed(2)
    # A KV entry: 128 NoPE channels in 64-wide groups, 64 RoPE channels kept.
    kv = torch.randn(2, 3, 192, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(triton(kv, rope_features=64), eager(kv, rope_features=64), rtol=0, atol=0)

    # An indexer activation: the whole 128-wide head, no RoPE split.
    act = torch.randn(2, 3, 128, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(triton(act), eager(act), rtol=0, atol=0)


@_TRITON_DEVICE
def test_triton_straight_through_backward_matches_eager(device: str):
    """The backward is the identity for both rows; a kernel that dropped it would show here."""
    _triton_op("standard", device)

    for impl in ("eager", "triton"):
        op = VeomniOp("kv_round", "standard", impl)
        torch.manual_seed(3)
        x = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        y = op(x, block_size=64)
        grad = torch.randn_like(y)
        (grad_x,) = torch.autograd.grad(y, x, grad)
        torch.testing.assert_close(grad_x, grad, rtol=0, atol=0, msg=impl)


def test_fp4_registry_entries():
    assert OP_REGISTRY.list_registered("kv_round", "fp4") == ["none", "eager", "triton"]
    assert OP_REGISTRY.default_impl("kv_round", "fp4", device="cpu") == "eager"


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("scale_fmt", ["e4m3", "ue8m0"])
def test_fp4_eager_round_trip_and_straight_through_backward(block_size: int, scale_fmt: str):
    op = VeomniOp("kv_round", "fp4", "eager")
    x = torch.randn(2, 4, block_size * 2, dtype=torch.bfloat16, requires_grad=True)
    y = op(x, block_size=block_size, scale_fmt=scale_fmt)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not torch.equal(y, x)

    grad_out = torch.randn_like(y)
    (grad_x,) = torch.autograd.grad(y, x, grad_out)
    torch.testing.assert_close(grad_x, grad_out)


@pytest.mark.parametrize("scale_fmt", ["e4m3", "ue8m0"])
def test_fp4_round_trip_is_idempotent(scale_fmt: str):
    """A second round trip must not move any value.

    That is the property a stored cache has to have: the tensor training sees is
    already on the E2M1 grid, so an encoder that round-trips its own output has to
    return it bit for bit. It also pins the nibble/midpoint convention -- a value
    sitting on a quantization boundary that drifted between calls would show up
    here as a different result on the second pass.

    Note the round trip is *not* required to stay under the input's amax: an E4M3
    scale is not a power of two, so the largest stored magnitude can be slightly
    above the block's own maximum. That magnitude is still a value the deployment
    kernel can represent, which is what ``E2M1 tops out at 6`` means.
    """
    from veomni.ops.kernels.kv_round.fp4.eager import fp4_roundtrip

    x = torch.randn(4, 8, 32, dtype=torch.bfloat16) * 3.0
    once = fp4_roundtrip(x, block_size=32, scale_fmt=scale_fmt)
    twice = fp4_roundtrip(once, block_size=32, scale_fmt=scale_fmt)

    torch.testing.assert_close(twice, once, rtol=0, atol=0)
    assert not torch.equal(once, x)


@pytest.mark.parametrize("scale_fmt", ["e4m3", "ue8m0"])
def test_fp4_saturates_at_six_times_the_block_scale(scale_fmt: str):
    """E2M1 has a hard top of 6, so a block of huge values clamps rather than wraps."""
    from veomni.ops.kernels.kv_round.fp4.eager import fp4_roundtrip

    x = torch.full((1, 2, 32), 1e4, dtype=torch.bfloat16)
    y = fp4_roundtrip(x, block_size=32, scale_fmt=scale_fmt)
    # A saturated block stores one identical magnitude in every slot.
    assert torch.isfinite(y).all()
    assert (y == y[..., :1]).all()


def test_fp4_rejects_an_unknown_scale_format():
    op = VeomniOp("kv_round", "fp4", "eager")
    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="scale_fmt must be one of"):
        op(x, block_size=32, scale_fmt="fp8")


def test_fp4_rejects_a_partial_group():
    op = VeomniOp("kv_round", "fp4", "eager")
    x = torch.randn(2, 4, 40, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not divisible by FP4 group size"):
        op(x, block_size=32)


@pytest.mark.parametrize("scale_fmt", ["e4m3", "ue8m0"])
def test_fp4_none_row_leaves_the_tensor_untouched(scale_fmt: str):
    """The off row must accept the same keywords the rounding rows do.

    It is the default ``kv_round_implementation``, so a caller that selects the
    fp4 variant and passes its documented ``scale_fmt`` reaches this row first. A
    shared ``none`` from another variant would raise here instead of doing
    nothing, which is why fp4 owns one.
    """
    op = VeomniOp("kv_round", "fp4", "none")
    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    assert torch.equal(op(x, block_size=32, scale_fmt=scale_fmt), x)


@_TRITON_DEVICE
@pytest.mark.parametrize("scale_fmt", ["e4m3", "ue8m0"])
@pytest.mark.parametrize("block_size", [16, 32])
def test_triton_fp4_matches_eager(device: str, scale_fmt: str, block_size: int):
    """The FP4 kernel must reproduce the eager round trip exactly, not approximately.

    ``e4m3`` is where the two can drift: its amax floor is ``6 * 2^-9`` and it
    divides by 6 *before* the narrowing cast, so an implementation that floors
    the scale instead of the amax hands back a value up to 6x too large for every
    small-magnitude block -- the common case once activations are normalized.
    The tiny and all-zero blocks below are what pin that down.
    """
    eager = VeomniOp("kv_round", "fp4", "eager")
    triton = _triton_op("fp4", device)

    torch.manual_seed(7)
    normal = torch.randn(3, 5, block_size * 2, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(
        triton(normal, block_size=block_size, scale_fmt=scale_fmt),
        eager(normal, block_size=block_size, scale_fmt=scale_fmt),
        rtol=0,
        atol=0,
        msg="normal",
    )

    tiny = torch.randn(2, 3, block_size, dtype=torch.bfloat16, device=device) * 1e-3
    torch.testing.assert_close(
        triton(tiny, block_size=block_size, scale_fmt=scale_fmt),
        eager(tiny, block_size=block_size, scale_fmt=scale_fmt),
        rtol=0,
        atol=0,
        msg="tiny",
    )

    zeros = torch.zeros(1, 2, block_size, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(
        triton(zeros, block_size=block_size, scale_fmt=scale_fmt),
        eager(zeros, block_size=block_size, scale_fmt=scale_fmt),
        rtol=0,
        atol=0,
        msg="zeros",
    )

    huge = torch.full((1, 2, block_size), 1e4, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(
        triton(huge, block_size=block_size, scale_fmt=scale_fmt),
        eager(huge, block_size=block_size, scale_fmt=scale_fmt),
        rtol=0,
        atol=0,
        msg="saturating",
    )


@pytest.mark.parametrize("amax", [0.0, 0.005, 0.02, 0.05, 0.2])
def test_fp4_e4m3_scale_floors_the_amax_not_the_scale(amax: float):
    """A block never stores a magnitude far above its own amax.

    The floor is applied to the amax (``6 * 2^-9``) *before* dividing by 6, so a
    sub-floor block gets a scale no larger than the floor's and the ratio it
    implies stays near ``6``. Flooring the *scale* instead hands back a scale up to
    6x bigger, and since the numerator is unchanged the stored magnitudes inflate
    with it -- a silent eager/Triton divergence for exactly the small blocks that
    normalized activations produce.

    Some overshoot is expected and is not the bug: near the floor an E4M3 scale
    sits in its subnormal range, where the step is coarse, so ``6 * scale`` can
    exceed the amax by ~17% for the smallest blocks. ``6x`` is a different
    failure entirely, and it is what flooring the scale instead of the amax
    produces.
    """
    from veomni.ops.kernels.kv_round.fp4.eager import fp4_roundtrip

    x = torch.full((1, 1, 32), amax, dtype=torch.bfloat16)
    y = fp4_roundtrip(x, block_size=32, scale_fmt="e4m3").float()

    assert torch.isfinite(y).all()
    assert y.abs().max() <= amax * 1.25 + 1e-6, f"stored {y.abs().max()} above amax {amax}"
    if amax == 0.0:
        assert torch.equal(y, torch.zeros_like(y))
