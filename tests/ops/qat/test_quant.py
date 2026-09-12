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

"""DeepSeek-V4 TileLang activation and weight quantization numerics."""

import pytest
import torch

from tests.ops.qat.reference import reference_act_quant, reference_fp8_weight_quant
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("DeepSeek V4 TileLang kernels require an NVIDIA CUDA GPU")
    if get_gpu_compute_capability() < 90:
        pytest.skip("DeepSeek V4 TileLang kernels require SM90 or later")


def test_tilelang_act_quant_shapes_scales_and_dequant():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import act_quant

    torch.manual_seed(2)
    x = torch.randn(3, 256, device=DEVICE, dtype=torch.bfloat16)
    quantized, scales = act_quant(x, block_size=128)

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 2)
    expected_quantized, expected_scales = reference_act_quant(x, block_size=128)
    assert torch.equal(scales, expected_scales)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    expected_dequantized = reference_act_quant(x, block_size=128, dequant=True)
    actual_dequantized = (quantized.float() * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
    torch.testing.assert_close(actual_dequantized, expected_dequantized, rtol=0, atol=0)

    result = act_quant(x, block_size=128, dequant=True)
    torch.testing.assert_close(result, expected_dequantized, rtol=0, atol=0)

    x_mx = x.clone()
    x_mx[0].zero_()
    quantized_mx, scales_mx = act_quant(
        x_mx,
        block_size=128,
        scale_fmt="ue8m0",
        scale_dtype=torch.float8_e8m0fnu,
    )
    expected_quantized_mx, expected_scales_mx = reference_act_quant(
        x_mx,
        block_size=128,
        scale_fmt="ue8m0",
        scale_dtype=torch.float8_e8m0fnu,
    )
    assert torch.equal(scales_mx.view(torch.uint8), expected_scales_mx.view(torch.uint8))
    assert torch.equal(quantized_mx.view(torch.uint8), expected_quantized_mx.view(torch.uint8))


def test_tilelang_act_quant_dequant_fuses_the_round_trip():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import act_quant

    torch.manual_seed(2)
    x = torch.randn(3, 256, device=DEVICE, dtype=torch.bfloat16)
    x[0].zero_()

    for scale_fmt in (None, "ue8m0"):
        quantized, scales = act_quant(x, block_size=128, scale_fmt=scale_fmt)
        expected = (quantized.float() * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
        original = x.clone()
        result = act_quant(x, block_size=128, scale_fmt=scale_fmt, dequant=True)

        assert result.dtype == torch.bfloat16
        assert torch.equal(result, expected)
        assert result.data_ptr() != x.data_ptr()
        assert torch.equal(x, original)

    transposed = torch.randn(256, 3, device=DEVICE, dtype=torch.bfloat16).t()
    assert not transposed.is_contiguous()
    assert torch.equal(
        act_quant(transposed, block_size=128, scale_fmt="ue8m0", dequant=True),
        act_quant(transposed.contiguous(), block_size=128, scale_fmt="ue8m0", dequant=True),
    )


def _weight_quant_test_input():
    """A 2x3 grid of 128x128 tiles, each exercising a different scale regime."""
    torch.manual_seed(6)
    tiles = [
        torch.randn(128, 128),
        torch.zeros(128, 128),  # amax clamp: must not divide by zero
        torch.full((128, 128), 1e-8),  # below the clamp floor
        torch.randn(128, 128) * 1e4,  # large magnitudes
        torch.randn(128, 128) * 1e-3,  # small magnitudes
        torch.randn(128, 128),
    ]
    tiles[4][7, 11] = 500.0  # lone outlier: the tile scale must absorb it
    rows = [torch.cat(tiles[:3], dim=1), torch.cat(tiles[3:], dim=1)]
    return torch.cat(rows, dim=0).to(device=DEVICE, dtype=torch.bfloat16)


def _tiles_above_amax_floor():
    """Mask of the _weight_quant_test_input tiles whose amax clears the 1e-4 floor.

    The other two tiles keep the floor scale, so they neither saturate nor
    stretch to the FP8 range and have to be excluded from range assertions.
    """
    return torch.tensor([[True, False, False], [True, True, True]], device=DEVICE)


def test_tilelang_fp8_weight_quant_matches_reference():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = _weight_quant_test_input()
    quantized, scales = fp8_weight_quant(x, block_size=128)
    reference_quantized, reference_scales = reference_fp8_weight_quant(x)

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    # A non-square tile grid catches a swapped block index or transposed scales.
    assert scales.shape == (2, 3)
    assert scales.dtype == torch.float32
    assert torch.equal(quantized.view(torch.uint8), reference_quantized.view(torch.uint8))
    assert torch.equal(scales, reference_scales)
    assert not quantized.float().isnan().any()

    # The scale is derived per tile, so each tile stretches to the FP8 max on
    # its own. The two tiles whose amax falls under the 1e-4 clamp keep the
    # floor scale instead and therefore stay far below saturation.
    tile_amax = quantized.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    assert torch.equal(tile_amax == 448.0, _tiles_above_amax_floor())
    assert tile_amax[0, 1] == 0.0
    assert 0.0 < tile_amax[0, 2] < 1.0
    assert scales[0, 1] == 1e-4 / 448.0
    assert scales[0, 2] == 1e-4 / 448.0


def test_tilelang_fp8_weight_quant_round_trip_and_non_contiguous_input():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = _weight_quant_test_input()
    quantized, scales = fp8_weight_quant(x, block_size=128)

    dequantized = quantized.float().view(2, 128, 3, 128) * scales[:, None, :, None]
    dequantized = dequantized.view(x.shape)
    # E4M3 keeps 3 mantissa bits, so a correctly scaled tile round-trips to
    # within one ulp (~2^-4) relative to that tile's amax.
    tile_amax = x.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    tolerance = (tile_amax / 16.0)[:, None, :, None].expand(2, 128, 3, 128).reshape(x.shape)
    assert ((dequantized - x.float()).abs() <= tolerance).all()

    transposed = x.t().contiguous().t()
    assert not transposed.is_contiguous()
    transposed_quantized, transposed_scales = fp8_weight_quant(transposed, block_size=128)
    assert torch.equal(transposed_quantized.view(torch.uint8), quantized.view(torch.uint8))
    assert torch.equal(transposed_scales, scales)


def test_tilelang_fp8_weight_quant_ue8m0_matches_reference():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = _weight_quant_test_input()
    quantized, scales = fp8_weight_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    reference_quantized, reference_scales = reference_fp8_weight_quant(
        x,
        scale_fmt="ue8m0",
        scale_dtype=torch.float8_e8m0fnu,
    )

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 3)
    assert scales.dtype == torch.float8_e8m0fnu
    # E8M0 stores the exponent alone, so a power-of-two scale survives the cast
    # bit for bit and matches the FP32 value the kernel divided by.
    assert torch.equal(scales.float().log2(), scales.float().log2().round())
    assert torch.equal(scales.float(), reference_scales)
    assert torch.equal(quantized.view(torch.uint8), reference_quantized.view(torch.uint8))
    assert not quantized.float().isnan().any()

    # Rounding the scale up costs at most one binade of range, so unlike the
    # FP32 mode no tile saturates, yet every tile above the amax floor still
    # reaches at least half of the FP8 max.
    tile_amax = quantized.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    assert (tile_amax <= 448.0).all()
    assert (tile_amax[_tiles_above_amax_floor()] >= 224.0).all()


def test_tilelang_fp8_weight_quant_scale_fmt_and_scale_dtype_are_orthogonal():
    """scale_fmt decides how the scale is computed, scale_dtype only how it is stored."""
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = _weight_quant_test_input()
    _, exact_scales = fp8_weight_quant(x, block_size=128)
    rounded_quantized, rounded_scales = fp8_weight_quant(x, block_size=128, scale_fmt="ue8m0")
    e8m0_quantized, e8m0_scales = fp8_weight_quant(
        x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu
    )

    assert rounded_scales.dtype == torch.float32
    assert torch.equal(rounded_scales, e8m0_scales.float())
    assert torch.equal(rounded_quantized.view(torch.uint8), e8m0_quantized.view(torch.uint8))
    # Rounding goes upward and never overshoots by a full binade.
    assert (rounded_scales >= exact_scales).all()
    assert (rounded_scales < 2.0 * exact_scales).all()


def test_tilelang_fp8_weight_quant_ue8m0_keeps_exact_power_of_two_scale():
    """An amax that already divides to a power of two must not gain a binade."""
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = torch.zeros(128, 128, device=DEVICE, dtype=torch.bfloat16)
    x[3, 5] = 56.0  # 448 * 2**-3, exact in BF16, so ceil(log2(amax / 448)) == -3
    quantized, scales = fp8_weight_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)

    assert scales.shape == (1, 1)
    assert scales.float().item() == 0.125
    assert quantized.float()[3, 5] == 448.0


def test_tilelang_fp8_weight_quant_ue8m0_round_trip():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = _weight_quant_test_input()
    quantized, scales = fp8_weight_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)

    dequantized = quantized.float().view(2, 128, 3, 128) * scales.float()[:, None, :, None]
    dequantized = dequantized.view(x.shape)
    # Rounding the scale up spends up to one of E4M3's 3 mantissa bits, so the
    # round-trip budget is twice the ~2^-4 of the exact-scale mode.
    tile_amax = x.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    tolerance = (tile_amax / 8.0)[:, None, :, None].expand(2, 128, 3, 128).reshape(x.shape)
    assert ((dequantized - x.float()).abs() <= tolerance).all()


def test_tilelang_fp8_weight_quant_dequant_fuses_the_round_trip():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    x = _weight_quant_test_input()
    rows, cols = x.shape

    for scale_fmt in (None, "ue8m0"):
        quantized, scales = fp8_weight_quant(x, block_size=128, scale_fmt=scale_fmt)
        tiles = quantized.float().view(rows // 128, 128, cols // 128, 128)
        expected = (tiles * scales[:, None, :, None]).view(rows, cols).to(torch.bfloat16)
        original = x.clone()
        result = fp8_weight_quant(x, block_size=128, scale_fmt=scale_fmt, dequant=True)

        assert result.dtype == torch.bfloat16
        assert torch.equal(result, expected)
        assert result.data_ptr() != x.data_ptr()
        assert torch.equal(x, original)

    transposed = x.t().contiguous().t()
    assert not transposed.is_contiguous()
    assert torch.equal(
        fp8_weight_quant(transposed, block_size=128, scale_fmt="ue8m0", dequant=True),
        fp8_weight_quant(x, block_size=128, scale_fmt="ue8m0", dequant=True),
    )


def test_tilelang_fp8_weight_quant_rejects_unsupported_inputs():
    _require_tilelang_cuda()
    from veomni.ops.qat.quant import fp8_weight_quant

    with pytest.raises(AssertionError, match="2D weight"):
        fp8_weight_quant(torch.empty(2, 128, 128, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(AssertionError, match="bfloat16 weight"):
        fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.float32))
    with pytest.raises(AssertionError, match="divisible by block_size"):
        fp8_weight_quant(torch.empty(128, 200, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(AssertionError, match="float32 and float8_e8m0fnu"):
        fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.bfloat16), scale_dtype=torch.float16)
    # An E8M0 scale cannot represent the unrounded FP32 scale the kernel would
    # divide by, so the pairing has to be rejected instead of drifting.
    with pytest.raises(AssertionError, match="powers of two"):
        fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.bfloat16), scale_dtype=torch.float8_e8m0fnu)
