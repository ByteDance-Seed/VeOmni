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

import importlib
import sys

import pytest
import torch
from torch import nn

from veomni.ops.qat import (
    fp8_blockwise,
    fp8_fake_quant_act,
    fp8_fake_quant_act_prefix,
    fp8_fake_quant_weight,
    qat_linear,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()
FP8_MAX = 448.0


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("DeepSeek V4 TileLang kernels require an NVIDIA CUDA GPU")
    if get_gpu_compute_capability() < 90:
        pytest.skip("DeepSeek V4 TileLang kernels require SM90 or later")


FP8_MAX_INV = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)


def _reference_scale(amax, scale_fmt):
    """The kernels' scale arithmetic, reproduced operation for operation.

    `amax * fl(1/448)` is not `amax / 448`: the reciprocal is rounded first, and
    the two disagree by one ulp for a bit over half of all inputs. The pow2 path
    is insensitive to that, but the unrounded path is not, so the stand-in has
    to multiply rather than divide.

    `fast_round_scale` (quant.py:75) reads the exponent straight off the
    FP32 bit pattern instead of calling `log2`/`ceil`, so do the same here
    rather than rely on the library's rounding agreeing.
    """
    scaled = amax * FP8_MAX_INV.to(amax.device)
    if not scale_fmt:
        return scaled
    bits = scaled.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    log2_ceil = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def _reference_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, dequant=False):
    """Torch stand-in for `veomni.ops.kernels.deepseek_v4.act_quant`.

    Reproduces the contract, not just the numbers: `dequant=True` fuses the
    dequantizing FP32 product and returns a fresh tensor, reading its operand
    without writing to it.
    """
    features = x.shape[-1]
    assert features % block_size == 0
    blocks = x.float().reshape(-1, features // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scales = _reference_scale(amax, scale_fmt)
    quantized = (blocks / scales).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if dequant:
        return (quantized.float() * scales).reshape(x.shape).to(x.dtype)
    return (
        quantized.reshape(x.shape),
        scales.reshape(*x.shape[:-1], features // block_size).to(scale_dtype),
    )


def _reference_fp8_weight_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, dequant=False):
    """Torch stand-in for `veomni.ops.kernels.deepseek_v4.fp8_weight_quant`.

    Reproduces the contract, not just the numbers: ``dequant=True`` fuses the
    dequantizing FP32 product and returns a fresh BF16 tensor, reading its
    operand without writing to it.
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    rows, cols = x.shape
    tiles = x.float().contiguous().view(rows // block_size, block_size, cols // block_size, block_size)
    amax = tiles.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scales = _reference_scale(amax, scale_fmt)
    quantized = (tiles / scales[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if dequant:
        return (quantized.float() * scales[:, None, :, None]).view(rows, cols).to(x.dtype)
    return quantized.view(rows, cols), scales.to(scale_dtype)


@pytest.fixture
def reference_quantizers(monkeypatch):
    """Swap the SM90 TileLang quantizers for the torch stand-ins above.

    Everything `fp8_blockwise` adds on top of the kernel call is hardware
    independent -- the copy that shields the caller's tensor from the in-place
    write, the straight-through gradient, the tile dequantization layout, the
    parameter substitution -- and that is what the tests using this fixture
    cover. `test_reference_quantizers_match_the_tilelang_kernels` pins the
    stand-ins to the real kernels wherever those can run.
    """
    monkeypatch.setattr(fp8_blockwise, "act_quant", _reference_act_quant)
    monkeypatch.setattr(fp8_blockwise, "fp8_weight_quant", _reference_fp8_weight_quant)


def _act_qdq(x, block_size=128, round_scale=True):
    return _reference_act_quant(x, block_size, "ue8m0" if round_scale else None, dequant=True)


def _weight_qdq(weight, block_size=128, round_scale=True):
    quantized, scales = _reference_fp8_weight_quant(weight, block_size, "ue8m0" if round_scale else None)
    rows, cols = weight.shape
    tiles = quantized.float().view(rows // block_size, block_size, cols // block_size, block_size)
    return (tiles * scales[:, None, :, None]).view(rows, cols).to(weight.dtype)


class _GroupedLinear(nn.Linear):
    """Stand-in for DeepSeek-V4's grouped output projection.

    `qat_linear` must route through the module's own forward rather than assume
    `F.linear`, and this is the shape of layer that catches the difference: the
    weight is reshaped into per-group blocks and applied with a `bmm`.
    """

    def __init__(self, in_features_per_group, out_features, n_groups):
        super().__init__(in_features_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x):
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        grouped = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        return torch.bmm(grouped, w).transpose(0, 1).reshape(*x.shape[:-2], self.n_groups, -1)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_qat_package_does_not_import_tilelang_eagerly():
    if "tilelang" in sys.modules:
        # Another test in this session already paid for the import, so a
        # re-import here could not tell us anything.
        pytest.skip("tilelang is already imported")
    original = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith("veomni.ops.qat")}
    try:
        importlib.import_module("veomni.ops.qat")
        assert "tilelang" not in sys.modules
    finally:
        sys.modules.update(original)


def test_fake_quant_rejects_non_bfloat16_operands():
    # The TileLang kernels hard-code a BF16 operand, so a FP32 tensor would be
    # reinterpreted rather than converted.
    with pytest.raises(TypeError, match="bfloat16 activation"):
        fp8_fake_quant_act(torch.zeros(2, 128))
    with pytest.raises(TypeError, match="bfloat16 weight"):
        fp8_fake_quant_weight(torch.zeros(128, 128))
    with pytest.raises(TypeError, match="bfloat16 activation"):
        fp8_fake_quant_act_prefix(torch.zeros(2, 512), 128)


def test_fake_quant_rejects_an_unknown_scale_fmt():
    # The kernels only test `scale_fmt is not None`, so a typo would silently
    # select power-of-two scales rather than fail.
    for call in (
        lambda fmt: fp8_fake_quant_act(torch.zeros(2, 128, dtype=torch.bfloat16), scale_fmt=fmt),
        lambda fmt: fp8_fake_quant_weight(torch.zeros(128, 128, dtype=torch.bfloat16), scale_fmt=fmt),
        lambda fmt: fp8_fake_quant_act_prefix(torch.zeros(2, 512, dtype=torch.bfloat16), 128, scale_fmt=fmt),
    ):
        with pytest.raises(ValueError, match="scale_fmt must be None or 'ue8m0'"):
            call("ue8m1")
        with pytest.raises(ValueError, match="scale_fmt must be None or 'ue8m0'"):
            call("UE8M0")


def test_fake_quant_weight_rejects_unsupported_shapes():
    with pytest.raises(ValueError, match="2D weight"):
        fp8_fake_quant_weight(torch.zeros(2, 128, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="divisible by block_size"):
        fp8_fake_quant_weight(torch.zeros(128, 200, dtype=torch.bfloat16))


def test_fake_quant_act_rejects_a_ragged_block_split():
    with pytest.raises(ValueError, match="activation of 200 channels is not divisible by block_size 128"):
        fp8_fake_quant_act(torch.zeros(2, 200, dtype=torch.bfloat16))
    # The prefix form names the argument the caller actually passed rather than
    # letting the inner call complain about a channel count it never saw.
    with pytest.raises(ValueError, match="quant_features 100 is not divisible by block_size 64"):
        fp8_fake_quant_act_prefix(torch.zeros(2, 512, dtype=torch.bfloat16), 100, block_size=64)


def test_fake_quant_act_prefix_rejects_out_of_range_split():
    x = torch.zeros(2, 512, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"\(0, 512\]"):
        fp8_fake_quant_act_prefix(x, 0)
    with pytest.raises(ValueError, match=r"\(0, 512\]"):
        fp8_fake_quant_act_prefix(x, 576)


# ---------------------------------------------------------------------------
# Wrapper behaviour, on the torch stand-ins
# ---------------------------------------------------------------------------


def test_fake_quant_act_dequantizes_in_place_of_its_input(reference_quantizers):
    torch.manual_seed(3)
    x = torch.randn(5, 3, 256, dtype=torch.bfloat16)

    torch.testing.assert_close(fp8_fake_quant_act(x), _act_qdq(x), rtol=0, atol=0)
    torch.testing.assert_close(
        fp8_fake_quant_act(x, block_size=64, scale_fmt=None),
        _act_qdq(x, block_size=64, round_scale=False),
        rtol=0,
        atol=0,
    )


def test_fake_quant_act_leaves_its_input_untouched(reference_quantizers):
    # The wrapper asks for `dequant=True` precisely because the in-place mode
    # would overwrite the tensor it is handed, and that tensor is the graph's own
    # activation -- every other consumer of it would read quantized values.
    torch.manual_seed(4)
    x = torch.randn(4, 256, dtype=torch.bfloat16, requires_grad=True)
    original = x.detach().clone()

    quantized = fp8_fake_quant_act(x)

    assert torch.equal(x.detach(), original)
    assert quantized.data_ptr() != x.data_ptr()
    assert not torch.equal(quantized.detach(), original)


def test_fake_quant_act_accepts_a_non_contiguous_input(reference_quantizers):
    # `fp8_fake_quant_act_prefix` slices the last dimension, and attention hands
    # over transposed head layouts, so neither caller can promise contiguity.
    torch.manual_seed(5)
    x = torch.randn(4, 8, 128, dtype=torch.bfloat16).transpose(0, 1)
    assert not x.is_contiguous()

    torch.testing.assert_close(fp8_fake_quant_act(x), _act_qdq(x), rtol=0, atol=0)


def test_fake_quant_act_passes_the_gradient_straight_through(reference_quantizers):
    torch.manual_seed(6)
    x = torch.randn(4, 256, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(4, 256, dtype=torch.bfloat16)

    fp8_fake_quant_act(x).backward(grad)

    assert torch.equal(x.grad, grad)


def test_fake_quant_act_prefix_quantizes_only_the_leading_slice(reference_quantizers):
    # DeepSeek-V4 stores the NoPE half of a KV entry in FP8 and keeps the
    # trailing RoPE channels in BF16.
    torch.manual_seed(7)
    x = torch.randn(2, 7, 512, dtype=torch.bfloat16)

    actual = fp8_fake_quant_act_prefix(x, quant_features=448, block_size=64)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual[..., :448], _act_qdq(x[..., :448], block_size=64), rtol=0, atol=0)
    assert torch.equal(actual[..., 448:], x[..., 448:])
    assert not torch.equal(actual[..., :448], x[..., :448])


def test_fake_quant_act_prefix_covering_every_channel_is_the_plain_quantizer(reference_quantizers):
    torch.manual_seed(8)
    x = torch.randn(3, 128, dtype=torch.bfloat16)

    actual = fp8_fake_quant_act_prefix(x, quant_features=128, block_size=128)

    torch.testing.assert_close(actual, fp8_fake_quant_act(x, block_size=128), rtol=0, atol=0)


def test_fake_quant_act_prefix_keeps_the_gradient_of_both_halves(reference_quantizers):
    torch.manual_seed(9)
    x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(2, 512, dtype=torch.bfloat16)

    fp8_fake_quant_act_prefix(x, quant_features=448, block_size=64).backward(grad)

    assert torch.equal(x.grad, grad)


def test_fake_quant_weight_dequantizes_each_tile_with_its_own_scale(reference_quantizers):
    torch.manual_seed(10)
    # A non-square tile grid catches a swapped block index or transposed scales.
    weight = torch.randn(256, 384, dtype=torch.bfloat16)

    actual = fp8_fake_quant_weight(weight)

    assert actual.shape == weight.shape
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, _weight_qdq(weight), rtol=0, atol=0)


def test_fake_quant_weight_leaves_its_input_untouched(reference_quantizers):
    # The tensor reaching this wrapper is a live parameter, so the fused
    # quantizer must read it and write elsewhere. Getting this wrong would
    # rewrite the master weight mid-step rather than fail.
    torch.manual_seed(10)
    weight = torch.randn(128, 256, dtype=torch.bfloat16, requires_grad=True)
    original = weight.detach().clone()

    quantized = fp8_fake_quant_weight(weight)

    assert torch.equal(weight.detach(), original)
    assert quantized.data_ptr() != weight.data_ptr()
    assert not torch.equal(quantized.detach(), original)


def test_fake_quant_weight_accepts_a_non_contiguous_weight(reference_quantizers):
    # A transposed weight reaches the quantizer from layers that store their
    # projection the other way round; the clone has to normalize the layout
    # rather than let the kernel reinterpret the strides.
    torch.manual_seed(10)
    weight = torch.randn(256, 128, dtype=torch.bfloat16).t()
    assert not weight.is_contiguous()

    torch.testing.assert_close(fp8_fake_quant_weight(weight), _weight_qdq(weight), rtol=0, atol=0)


def test_fake_quant_weight_passes_the_gradient_straight_through(reference_quantizers):
    torch.manual_seed(11)
    weight = torch.randn(128, 256, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(128, 256, dtype=torch.bfloat16)

    fp8_fake_quant_weight(weight).backward(grad)

    assert torch.equal(weight.grad, grad)


def test_fake_quant_weight_round_trips_within_one_e4m3_ulp(reference_quantizers):
    torch.manual_seed(12)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)

    dequantized = fp8_fake_quant_weight(weight).float()

    # Rounding the scale up to a power of two spends one of E4M3's 3 mantissa
    # bits, so the budget is twice the ~2^-4 of an exact scale.
    tolerance = weight.float().abs().amax() / 8.0
    assert ((dequantized - weight.float()).abs() <= tolerance).all()


def test_fake_quant_weight_scales_each_tile_independently(reference_quantizers):
    # One tile carrying a huge outlier must not drag the other tiles' scales
    # down, which is the whole reason the weight quantizer is 2D-blocked. Keep
    # the two magnitudes off the diagonal: a diagonal layout is invariant under
    # transposition, so it would not catch swapped tile indices.
    weight = torch.zeros(256, 256, dtype=torch.bfloat16)
    weight[:128, 128:] = 1e4
    weight[128:, :128] = 1e-2

    dequantized = fp8_fake_quant_weight(weight).float()

    # A single scale spanning both tiles would flush the 1e-2 tile to zero.
    assert (dequantized[128:, :128] > 0).all()
    # E4M3 keeps 3 mantissa bits and the scale is rounded up to a power of two,
    # so each tile survives to within about one binade's worth of ulp.
    torch.testing.assert_close(dequantized[:128, 128:], weight[:128, 128:].float(), rtol=0.1, atol=0)
    torch.testing.assert_close(dequantized[128:, :128], weight[128:, :128].float(), rtol=0.1, atol=0)
    assert (dequantized[:128, :128] == 0).all()


# ---------------------------------------------------------------------------
# qat_linear
# ---------------------------------------------------------------------------


def test_qat_linear_disabled_is_a_transparent_wrapper():
    # The whole point of the `enabled` flag: a converted call site stays valid,
    # and free, when QAT is off -- including on hosts without the kernels.
    torch.manual_seed(13)
    linear = nn.Linear(128, 64, bias=True, dtype=torch.bfloat16)
    x = torch.randn(4, 128, dtype=torch.bfloat16, requires_grad=True)

    actual = qat_linear(linear, x, enabled=False)

    assert torch.equal(actual, linear(x))
    actual.sum().backward()
    assert x.grad is not None


def test_qat_linear_refuses_a_module_that_does_not_own_its_weight():
    # `functional_call` substitutes by name and ignores names it cannot find, so
    # a delegating wrapper would quantize a tensor and then run the GEMM with
    # the original weight anyway. Silent no-op QAT has to be an error.
    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)

        @property
        def weight(self):
            return self.base_layer.weight

        def forward(self, x):
            return self.base_layer(x)

    with pytest.raises(TypeError, match="does not own a 'weight' parameter"):
        qat_linear(_Wrapper(), torch.zeros(2, 128, dtype=torch.bfloat16))


def test_qat_linear_composes_the_two_quantizers(reference_quantizers):
    torch.manual_seed(14)
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16)

    actual = qat_linear(linear, x)
    expected = nn.functional.linear(fp8_fake_quant_act(x), fp8_fake_quant_weight(linear.weight))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qat_linear_keeps_the_bias_out_of_the_quantizer(reference_quantizers):
    torch.manual_seed(15)
    linear = nn.Linear(256, 128, bias=True, dtype=torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16)

    actual = qat_linear(linear, x)
    expected = nn.functional.linear(fp8_fake_quant_act(x), fp8_fake_quant_weight(linear.weight), linear.bias)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qat_linear_weight_only_leaves_the_activation_alone(reference_quantizers):
    torch.manual_seed(16)
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16)

    actual = qat_linear(linear, x, quantize_activation=False)
    expected = nn.functional.linear(x, fp8_fake_quant_weight(linear.weight))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qat_linear_gradients_reach_the_real_parameter(reference_quantizers):
    torch.manual_seed(17)
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(8, 128, dtype=torch.bfloat16)

    qat_linear(linear, x).backward(grad)

    # Straight-through on both operands, so the surviving gradients are the
    # GEMM's own, computed against the *quantized* operands.
    quantized_weight = fp8_fake_quant_weight(linear.weight.detach())
    quantized_x = fp8_fake_quant_act(x.detach())
    torch.testing.assert_close(x.grad, grad @ quantized_weight)
    torch.testing.assert_close(linear.weight.grad, grad.t() @ quantized_x)


def test_qat_linear_preserves_a_non_functional_linear_forward(reference_quantizers):
    torch.manual_seed(18)
    grouped = _GroupedLinear(256, 512, n_groups=4).to(dtype=torch.bfloat16)
    x = torch.randn(6, 4, 256, dtype=torch.bfloat16)

    actual = qat_linear(grouped, x)

    # An `F.linear` shortcut inside `qat_linear` would not even produce this
    # shape, let alone these values.
    assert actual.shape == grouped(x).shape
    quantized_weight = fp8_fake_quant_weight(grouped.weight).view(4, -1, 256).transpose(1, 2)
    quantized_x = fp8_fake_quant_act(x).reshape(-1, 4, 256).transpose(0, 1)
    expected = torch.bmm(quantized_x, quantized_weight).transpose(0, 1).reshape(6, 4, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qat_linear_restores_the_parameter_after_the_call(reference_quantizers):
    torch.manual_seed(19)
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    original = linear.weight.detach().clone()

    qat_linear(linear, torch.randn(8, 256, dtype=torch.bfloat16))

    assert isinstance(linear.weight, nn.Parameter)
    assert torch.equal(linear.weight.detach(), original)


# ---------------------------------------------------------------------------
# Kernel parity: the only part that needs the hardware
# ---------------------------------------------------------------------------


def test_reference_quantizers_match_the_tilelang_kernels():
    """Pin the torch stand-ins the tests above rely on to the real kernels."""
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import act_quant, fp8_weight_quant

    torch.manual_seed(20)
    x = torch.randn(6, 256, device=DEVICE, dtype=torch.bfloat16)
    x[0].zero_()  # amax clamp floor: must not divide by zero
    x[1] *= 1e4
    weight = torch.randn(256, 384, device=DEVICE, dtype=torch.bfloat16)
    weight[:128, :128] = 0.0

    for block_size in (64, 128):
        for scale_fmt in (None, "ue8m0"):
            actual = act_quant(x, block_size, scale_fmt, dequant=True)
            expected = _reference_act_quant(x, block_size, scale_fmt, dequant=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            quantized, scales = act_quant(x, block_size, scale_fmt)
            reference_quantized, reference_scales = _reference_act_quant(x, block_size, scale_fmt)
            assert torch.equal(quantized.view(torch.uint8), reference_quantized.view(torch.uint8))
            assert torch.equal(scales, reference_scales)

    for scale_fmt in (None, "ue8m0"):
        quantized, scales = fp8_weight_quant(weight, 128, scale_fmt)
        reference_quantized, reference_scales = _reference_fp8_weight_quant(weight, 128, scale_fmt)
        assert torch.equal(quantized.view(torch.uint8), reference_quantized.view(torch.uint8))
        assert torch.equal(scales, reference_scales)


def test_qat_linear_runs_on_the_tilelang_kernels():
    _require_tilelang_cuda()
    torch.manual_seed(21)
    linear = nn.Linear(256, 128, bias=False, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(8, 256, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    actual = qat_linear(linear, x)
    actual.sum().backward()

    expected = nn.functional.linear(fp8_fake_quant_act(x.detach()), fp8_fake_quant_weight(linear.weight.detach()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert x.grad is not None and linear.weight.grad is not None
