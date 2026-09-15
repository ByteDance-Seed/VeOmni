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

"""FP8 fake-quantization and QAT linear contract tests."""

import sys
from types import ModuleType

import pytest
import torch
from torch import nn

import veomni.ops.qat as qat
from tests.ops.qat.reference import reference_act_quant, reference_fp8_weight_quant
from tests.ops.utils import require_nvidia_cuda
from veomni.ops.qat import _hardware as qat_hardware
from veomni.ops.qat import (
    fp8_blockwise,
    fp8_fake_quant_act,
    fp8_fake_quant_act_prefix,
    fp8_fake_quant_weight,
    qat_linear,
)
from veomni.utils.device import get_device_type


DEVICE = get_device_type()


@pytest.fixture(autouse=True)
def clear_qat_hardware_gate_cache():
    """Keep process-level hardware caching isolated between QAT tests."""
    qat_hardware.require_tilelang_sm90.cache_clear()
    yield
    qat_hardware.require_tilelang_sm90.cache_clear()


def _qat_entry_calls():
    activation = torch.zeros(1, 256, dtype=torch.bfloat16)
    fp4_weight = torch.zeros(2, 32, dtype=torch.bfloat16)
    fp8_weight = torch.zeros(128, 128, dtype=torch.bfloat16)
    stacked_weight = torch.zeros(1, 128, 128, dtype=torch.bfloat16)
    linear = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
    return (
        ("act_quant", lambda: qat.act_quant(activation)),
        ("fp4_act_quant", lambda: qat.fp4_act_quant(fp4_weight)),
        ("fp8_weight_quant", lambda: qat.fp8_weight_quant(fp8_weight)),
        ("fp4_fake_quant_weight", lambda: qat.fp4_fake_quant_weight(fp4_weight)),
        ("fp8_fake_quant_act", lambda: qat.fp8_fake_quant_act(activation)),
        ("fp8_fake_quant_act_prefix", lambda: qat.fp8_fake_quant_act_prefix(activation, 128)),
        ("fp8_fake_quant_weight", lambda: qat.fp8_fake_quant_weight(fp8_weight)),
        ("fp8_fake_quant_stacked_weight", lambda: qat.fp8_fake_quant_stacked_weight(stacked_weight)),
        ("qat_linear", lambda: qat.qat_linear(linear, activation[..., :128])),
    )


def _install_fake_quant_backend(monkeypatch, calls: list[str]) -> None:
    backend = ModuleType("veomni.ops.qat.quant")

    def record(name):
        def fake_quant(tensor, *args, **kwargs):
            calls.append(name)
            return tensor.clone()

        return fake_quant

    backend.act_quant = record("act_quant")
    backend.fp4_act_quant = record("fp4_act_quant")
    backend.fp8_weight_quant = record("fp8_weight_quant")
    monkeypatch.setitem(sys.modules, "veomni.ops.qat.quant", backend)


def test_qat_entry_points_reject_pre_sm90_before_vendor_import(monkeypatch):
    """Every quantizing QAT path checks hardware before entering TileLang."""
    vendor_calls = []
    _install_fake_quant_backend(monkeypatch, vendor_calls)
    monkeypatch.setattr(type(qat_hardware.NVIDIA_SM90_PLUS), "matches", lambda self: False)

    for name, call in _qat_entry_calls():
        with pytest.raises(RuntimeError, match="SM90 or later NVIDIA CUDA GPU"):
            call()
        assert vendor_calls == [], f"{name} entered the vendor backend before the hardware check"


def test_qat_entry_points_reach_vendor_backend_after_sm90_check(monkeypatch):
    """The hardware gate does not intercept supported QAT calls."""
    vendor_calls = []
    hardware_checks = []
    _install_fake_quant_backend(monkeypatch, vendor_calls)

    def supports_sm90(self):
        hardware_checks.append(True)
        return True

    monkeypatch.setattr(type(qat_hardware.NVIDIA_SM90_PLUS), "matches", supports_sm90)

    for name, call in _qat_entry_calls():
        vendor_calls.clear()
        call()
        assert vendor_calls, f"{name} did not reach the vendor backend after the hardware check"
    assert hardware_checks == [True]


@pytest.fixture
def reference_quantizers(monkeypatch):
    """Replace hardware-backed FP8 quantizers with independent torch oracles.

    This keeps facade behavior CPU-testable while preserving reference
    quantize-dequantize numerics for layout, prefix, straight-through gradients,
    and temporary parameter substitution assertions.
    """
    monkeypatch.setattr(fp8_blockwise, "act_quant", reference_act_quant)
    monkeypatch.setattr(fp8_blockwise, "fp8_weight_quant", reference_fp8_weight_quant)


def _act_qdq(x, block_size=128, round_scale=True):
    return reference_act_quant(x, block_size, "ue8m0" if round_scale else None, dequant=True)


def _weight_qdq(weight, block_size=128, round_scale=True):
    quantized, scales = reference_fp8_weight_quant(weight, block_size, "ue8m0" if round_scale else None)
    rows, cols = weight.shape
    tiles = quantized.float().view(rows // block_size, block_size, cols // block_size, block_size)
    return (tiles * scales[:, None, :, None]).view(rows, cols).to(weight.dtype)


def _fake_quant(kind, tensor):
    return fp8_fake_quant_act(tensor) if kind == "activation" else fp8_fake_quant_weight(tensor)


def _fake_quant_reference(kind, tensor):
    return _act_qdq(tensor) if kind == "activation" else _weight_qdq(tensor)


def _fake_quant_input(kind, *, non_contiguous=False, requires_grad=False):
    if kind == "activation":
        tensor = (
            torch.randn(4, 8, 128, dtype=torch.bfloat16).transpose(0, 1)
            if non_contiguous
            else torch.randn(4, 256, dtype=torch.bfloat16)
        )
    else:
        tensor = (
            torch.randn(256, 128, dtype=torch.bfloat16).t()
            if non_contiguous
            else torch.randn(128, 256, dtype=torch.bfloat16)
        )
    return tensor.requires_grad_(requires_grad)


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


@pytest.mark.parametrize("kind", ("activation", "weight"))
def test_fake_quant_leaves_its_input_untouched(reference_quantizers, kind):
    torch.manual_seed(4)
    tensor = _fake_quant_input(kind, requires_grad=True)
    original = tensor.detach().clone()

    quantized = _fake_quant(kind, tensor)

    assert torch.equal(tensor.detach(), original)
    assert quantized.data_ptr() != tensor.data_ptr()
    assert not torch.equal(quantized.detach(), original)


@pytest.mark.parametrize("kind", ("activation", "weight"))
def test_fake_quant_accepts_a_non_contiguous_operand(reference_quantizers, kind):
    torch.manual_seed(5)
    tensor = _fake_quant_input(kind, non_contiguous=True)
    assert not tensor.is_contiguous()

    torch.testing.assert_close(_fake_quant(kind, tensor), _fake_quant_reference(kind, tensor), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("activation", "weight"))
def test_fake_quant_passes_the_gradient_straight_through(reference_quantizers, kind):
    torch.manual_seed(6)
    tensor = _fake_quant_input(kind, requires_grad=True)
    grad = torch.randn_like(tensor)

    _fake_quant(kind, tensor).backward(grad)

    assert torch.equal(tensor.grad, grad)


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


@pytest.mark.parametrize(
    ("with_bias", "quantize_activation"),
    (
        (False, True),
        (True, True),
        (False, False),
    ),
    ids=("act-and-weight", "with-bias", "weight-only"),
)
def test_qat_linear_composes_requested_quantizers(reference_quantizers, with_bias, quantize_activation):
    torch.manual_seed(14)
    linear = nn.Linear(256, 128, bias=with_bias, dtype=torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16)

    actual = qat_linear(linear, x, quantize_activation=quantize_activation)
    quantized_x = fp8_fake_quant_act(x) if quantize_activation else x
    expected = nn.functional.linear(quantized_x, fp8_fake_quant_weight(linear.weight), linear.bias)

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


def test_qat_linear_runs_on_the_tilelang_kernels():
    require_nvidia_cuda("tilelang", min_cc=90)
    torch.manual_seed(21)
    linear = nn.Linear(256, 128, bias=False, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(8, 256, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    actual = qat_linear(linear, x)
    actual.sum().backward()

    expected = nn.functional.linear(fp8_fake_quant_act(x.detach()), fp8_fake_quant_weight(linear.weight.detach()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert x.grad is not None and linear.weight.grad is not None
