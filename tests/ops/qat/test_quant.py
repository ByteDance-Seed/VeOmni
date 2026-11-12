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

"""DeepSeek-V4 activation and weight quantization numerics.

FP8 tests run against each backend the host supports: the TileLang kernels on
SM90+ NVIDIA GPUs and their Triton port on AMD gfx942/gfx950. Both must match
the same Torch reference bit for bit.
"""

import pytest
import torch

from tests.ops.qat.reference import reference_act_quant, reference_fp4_act_quant, reference_fp8_weight_quant
from tests.ops.utils import require_nvidia_cuda
from veomni.ops.qat._hardware import TRITON_FP8_ARCHS
from veomni.utils.device import get_device_type, get_rocm_gcn_arch


DEVICE = get_device_type()


@pytest.fixture(params=("tilelang", "triton"))
def fp8_quantizers(request):
    """Module exposing ``act_quant`` / ``fp8_weight_quant`` for one backend."""
    if request.param == "tilelang":
        require_nvidia_cuda("tilelang", min_cc=90)
        from veomni.ops.qat import quant

        return quant
    pytest.importorskip("triton")
    if get_rocm_gcn_arch() not in TRITON_FP8_ARCHS:
        pytest.skip("Triton FP8 quantizers require an AMD gfx942/gfx950 GPU")
    from veomni.ops.qat import triton_quant

    return triton_quant


def _require_triton_quantizers():
    pytest.importorskip("triton")
    if get_rocm_gcn_arch() not in TRITON_FP8_ARCHS:
        pytest.skip("requires the ROCm Triton FP8 quantizers")
    from veomni.ops.qat import triton_quant

    return triton_quant


def test_fp8_scale_pairing_rejects_e8m0_without_format():
    from veomni.ops.qat.scale import validate_fp8_scale_pairing

    validate_fp8_scale_pairing(None, torch.float32)
    validate_fp8_scale_pairing("ue8m0", torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="powers of two"):
        validate_fp8_scale_pairing(None, torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="float32 and float8_e8m0fnu"):
        validate_fp8_scale_pairing(None, torch.float16)
    with pytest.raises(ValueError, match="scale_fmt"):
        validate_fp8_scale_pairing("typo", torch.float32)


def test_tilelang_fp4_quant_matches_reference_and_dequantizes():
    require_nvidia_cuda("tilelang", min_cc=90)
    from veomni.ops.qat.quant import fp4_act_quant

    torch.manual_seed(1)
    rounding_ties = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0)).repeat(8)
    x = torch.stack(
        (
            torch.zeros(64),
            torch.full((64,), 1e-8),
            torch.linspace(-7.0, 7.0, 64),
            rounding_ties,
            torch.randn(64),
        )
    ).to(device=DEVICE, dtype=torch.bfloat16)
    x[-1, 9] = 500.0
    quantized, scales = fp4_act_quant(x)
    reference_quantized, reference_scales = reference_fp4_act_quant(x)

    assert quantized.shape == (5, 32)
    assert quantized.dtype == torch.float4_e2m1fn_x2
    assert scales.shape == (5, 2)
    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(quantized.view(torch.uint8), reference_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), reference_scales.view(torch.uint8))

    dequantized = fp4_act_quant(x, dequant=True)
    expected_dequantized = reference_fp4_act_quant(x, dequant=True)
    assert dequantized.dtype == torch.bfloat16
    assert torch.equal(dequantized, expected_dequantized)


@pytest.mark.parametrize("block_size", (64, 128), ids=("block64", "block128"))
@pytest.mark.parametrize(
    ("scale_fmt", "scale_dtype"),
    ((None, torch.float32), ("ue8m0", torch.float8_e8m0fnu)),
    ids=("exact-scale", "ue8m0"),
)
def test_act_quant_matches_reference_and_dequantizes(fp8_quantizers, block_size, scale_fmt, scale_dtype):
    act_quant = fp8_quantizers.act_quant

    torch.manual_seed(2)
    x = torch.randn(256, 3, device=DEVICE, dtype=torch.bfloat16).t()
    x[0].zero_()
    assert not x.is_contiguous()
    original = x.clone()
    kwargs = {"block_size": block_size, "scale_fmt": scale_fmt, "scale_dtype": scale_dtype}
    quantized, scales = act_quant(x, **kwargs)

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 256 // block_size)
    assert scales.dtype == scale_dtype
    expected_quantized, expected_scales = reference_act_quant(x, **kwargs)
    if scale_dtype == torch.float8_e8m0fnu:
        assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    else:
        assert torch.equal(scales, expected_scales)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    dequantized = act_quant(x, dequant=True, **kwargs)
    expected_dequantized = reference_act_quant(x, dequant=True, **kwargs)
    assert dequantized.dtype == torch.bfloat16
    assert torch.equal(dequantized, expected_dequantized)
    assert dequantized.data_ptr() != x.data_ptr()
    assert torch.equal(x, original)


@pytest.mark.parametrize(
    ("scale_fmt", "scale_dtype"),
    ((None, torch.float32), ("ue8m0", torch.float8_e8m0fnu)),
    ids=("exact-scale", "ue8m0"),
)
def test_act_quant_matches_reference_across_row_tiles(fp8_quantizers, scale_fmt, scale_dtype):
    """Rows past a partial tile, mixed magnitudes and signed zeros all round like the reference."""
    torch.manual_seed(3)
    magnitudes = torch.logspace(-6, 4, 1001).unsqueeze(-1)
    x = (torch.randn(1001, 384) * magnitudes).to(device=DEVICE, dtype=torch.bfloat16)
    x[5, :128] = -0.0
    x[6, 7] = -1e-30
    kwargs = {"block_size": 128, "scale_fmt": scale_fmt, "scale_dtype": scale_dtype}

    quantized, scales = fp8_quantizers.act_quant(x, **kwargs)
    expected_quantized, expected_scales = reference_act_quant(x, **kwargs)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert torch.equal(
        fp8_quantizers.act_quant(x, dequant=True, **kwargs), reference_act_quant(x, dequant=True, **kwargs)
    )


@pytest.mark.parametrize("scale_fmt", (None, "ue8m0"))
def test_triton_act_quant_propagates_nonfinite_blocks_like_reference(scale_fmt):
    quantizers = _require_triton_quantizers()
    x = torch.zeros(3, 128, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0] = float("nan")
    x[1, 0] = float("inf")
    x[2, 0] = float("-inf")

    actual, actual_scales = quantizers.act_quant(x, scale_fmt=scale_fmt)
    expected, expected_scales = reference_act_quant(x, scale_fmt=scale_fmt)
    actual_float, expected_float = actual.float(), expected.float()
    assert torch.equal(torch.isnan(actual_float), torch.isnan(expected_float))
    assert torch.equal(torch.nan_to_num(actual_float), torch.nan_to_num(expected_float))
    assert torch.isnan(actual_scales[0]).all()
    assert torch.equal(actual_scales[1:], expected_scales[1:])

    actual_dequant = quantizers.act_quant(x, scale_fmt=scale_fmt, dequant=True)
    expected_dequant = reference_act_quant(x, scale_fmt=scale_fmt, dequant=True)
    assert torch.equal(torch.isnan(actual_dequant), torch.isnan(expected_dequant))
    assert torch.equal(torch.nan_to_num(actual_dequant), torch.nan_to_num(expected_dequant))


def test_triton_fp8_rounding_ties_and_subnormal_boundary_match_reference():
    quantizers = _require_triton_quantizers()
    x = torch.zeros(1, 128, device=DEVICE, dtype=torch.bfloat16)
    x[0, :6] = torch.tensor(
        [448.0, 1.0625, -1.0625, 2**-10, -(2**-10), -0.0],
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    actual, scales = quantizers.act_quant(x, scale_fmt="ue8m0")
    expected, expected_scales = reference_act_quant(x, scale_fmt="ue8m0")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))


def test_triton_quantizers_reject_invalid_public_inputs():
    quantizers = _require_triton_quantizers()
    with pytest.raises(ValueError, match="nonzero last dimension"):
        quantizers.act_quant(torch.empty(2, 0, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="scale_fmt"):
        quantizers.act_quant(torch.empty(2, 128, device=DEVICE, dtype=torch.bfloat16), scale_fmt="typo")
    with pytest.raises(ValueError, match="scale_fmt"):
        quantizers.fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.bfloat16), scale_fmt="typo")


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


def test_fp8_weight_quant_matches_reference(fp8_quantizers):
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

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


@pytest.mark.parametrize("scale_fmt", (None, "ue8m0"))
def test_triton_weight_quant_propagates_nonfinite_tile_like_reference(scale_fmt):
    quantizers = _require_triton_quantizers()
    x = torch.zeros(128, 128, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0] = float("nan")

    actual, actual_scales = quantizers.fp8_weight_quant(x, scale_fmt=scale_fmt)
    expected, _ = reference_fp8_weight_quant(x, scale_fmt=scale_fmt)
    actual_float, expected_float = actual.float(), expected.float()
    assert torch.equal(torch.isnan(actual_float), torch.isnan(expected_float))
    assert torch.equal(torch.nan_to_num(actual_float), torch.nan_to_num(expected_float))
    assert torch.isnan(actual_scales).all()

    actual_dequant = quantizers.fp8_weight_quant(x, scale_fmt=scale_fmt, dequant=True)
    expected_dequant = reference_fp8_weight_quant(x, scale_fmt=scale_fmt, dequant=True)
    assert torch.equal(torch.isnan(actual_dequant), torch.isnan(expected_dequant))


@pytest.mark.parametrize(
    ("scale_fmt", "scale_dtype", "tolerance_divisor"),
    (
        (None, None, 16.0),
        ("ue8m0", torch.float8_e8m0fnu, 8.0),
    ),
    ids=("exact-scale", "ue8m0"),
)
def test_fp8_weight_quant_round_trip_and_non_contiguous_input(
    fp8_quantizers, scale_fmt, scale_dtype, tolerance_divisor
):
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

    x = _weight_quant_test_input()
    kwargs = {"scale_fmt": scale_fmt}
    if scale_dtype is not None:
        kwargs["scale_dtype"] = scale_dtype
    quantized, scales = fp8_weight_quant(x, block_size=128, **kwargs)

    dequantized = quantized.float().view(2, 128, 3, 128) * scales.float()[:, None, :, None]
    dequantized = dequantized.view(x.shape)
    # E8M0 scale rounding spends one additional mantissa bit of the E4M3 value.
    tile_amax = x.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    tolerance = (tile_amax / tolerance_divisor)[:, None, :, None].expand(2, 128, 3, 128).reshape(x.shape)
    assert ((dequantized - x.float()).abs() <= tolerance).all()

    transposed = x.t().contiguous().t()
    assert not transposed.is_contiguous()
    transposed_quantized, transposed_scales = fp8_weight_quant(transposed, block_size=128, **kwargs)
    assert torch.equal(transposed_quantized.view(torch.uint8), quantized.view(torch.uint8))
    assert torch.equal(transposed_scales, scales)


def test_fp8_weight_quant_ue8m0_matches_reference(fp8_quantizers):
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

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
    assert reference_scales.dtype == torch.float8_e8m0fnu
    # E8M0 stores the exponent alone, so a power-of-two scale survives the cast
    # bit for bit and matches the FP32 value the kernel divided by.
    assert torch.equal(scales.float().log2(), scales.float().log2().round())
    assert torch.equal(scales.view(torch.uint8), reference_scales.view(torch.uint8))
    assert torch.equal(quantized.view(torch.uint8), reference_quantized.view(torch.uint8))
    assert not quantized.float().isnan().any()

    # Rounding the scale up costs at most one binade of range, so unlike the
    # FP32 mode no tile saturates, yet every tile above the amax floor still
    # reaches at least half of the FP8 max.
    tile_amax = quantized.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    assert (tile_amax <= 448.0).all()
    assert (tile_amax[_tiles_above_amax_floor()] >= 224.0).all()


def test_fp8_weight_quant_scale_fmt_and_scale_dtype_are_orthogonal(fp8_quantizers):
    """scale_fmt decides how the scale is computed, scale_dtype only how it is stored."""
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

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


def test_fp8_weight_quant_ue8m0_keeps_exact_power_of_two_scale(fp8_quantizers):
    """An amax that already divides to a power of two must not gain a binade."""
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

    x = torch.zeros(128, 128, device=DEVICE, dtype=torch.bfloat16)
    x[3, 5] = 56.0  # 448 * 2**-3, exact in BF16, so ceil(log2(amax / 448)) == -3
    quantized, scales = fp8_weight_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)

    assert scales.shape == (1, 1)
    assert scales.float().item() == 0.125
    assert quantized.float()[3, 5] == 448.0


def test_fp8_weight_quant_dequant_fuses_the_round_trip(fp8_quantizers):
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

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


def test_fp8_weight_quant_rejects_unsupported_inputs(fp8_quantizers):
    fp8_weight_quant = fp8_quantizers.fp8_weight_quant

    with pytest.raises(ValueError, match="2D weight"):
        fp8_weight_quant(torch.empty(2, 128, 128, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="bfloat16"):
        fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.float32))
    with pytest.raises(ValueError, match="divisible by block_size"):
        fp8_weight_quant(torch.empty(128, 200, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="float32 and float8_e8m0fnu"):
        fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.bfloat16), scale_dtype=torch.float16)
    # An E8M0 scale cannot represent the unrounded FP32 scale the kernel would
    # divide by, so the pairing has to be rejected instead of drifting.
    with pytest.raises(ValueError, match="powers of two"):
        fp8_weight_quant(torch.empty(128, 128, device=DEVICE, dtype=torch.bfloat16), scale_dtype=torch.float8_e8m0fnu)


def test_act_quant_rejects_e8m0_without_format(fp8_quantizers):
    with pytest.raises(ValueError, match="powers of two"):
        fp8_quantizers.act_quant(
            torch.empty(4, 128, device=DEVICE, dtype=torch.bfloat16), scale_dtype=torch.float8_e8m0fnu
        )
