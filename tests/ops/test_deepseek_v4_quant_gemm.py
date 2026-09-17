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

"""The quantized GEMMs that let DeepSeek-V4.1 run in the precision it ships in.

What these are tested against matters. The obvious reference -- the original
bf16 ``x @ w.T`` -- is the wrong one: it folds the quantizer's error into the
GEMM's, and the combined tolerance needed to pass would be wide enough to hide
a real kernel bug. The reference here is instead the *dequantized operands*
multiplied in float32, which is exactly what the kernel claims to compute, so
the only admissible difference is accumulation order.

That reference is exact rather than approximate. Both fp8_e4m3 (3 mantissa
bits) and fp4_e2m1 (1 mantissa bit) carry fewer bits than bf16's 7, and every
scale is an e8m0 power of two, so dequantization only shifts an exponent. The
bf16 round trip that ``dequant=True`` returns therefore loses nothing, and
casting it to float32 reproduces the kernel's own operands bit for bit.

So the kernels are checked at two levels, because a single tolerance would
measure the wrong thing. Asked for a float32 output, the kernel agrees with the
float32 reference to about 2e-8 -- it accumulates in float32 throughout, so
only summation order separates them, and that is the real gate. Asked for bf16,
the error jumps to 1.6e-3, which is not kernel error at all: it is bf16's half
ulp, and it matches rounding the float32 reference to bf16 to four digits. The
bf16 path is therefore held to "no worse than correct rounding" rather than to
a loose absolute number, which would have passed a kernel that was genuinely
three orders of magnitude wrong.
"""

import pytest
import torch

from veomni.ops.kernels.deepseek_v4 import act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, fp8_weight_quant
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()

requires_sm90 = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or torch.version.hip is not None or get_gpu_compute_capability() < 90,
    reason="DeepSeek-V4 quantized GEMMs are TileLang kernels needing an SM90 or later NVIDIA GPU",
)

#: The released checkpoint's scale format. E8M0 has no mantissa, so it is only
#: meaningful next to ``scale_fmt="ue8m0"``, which rounds scales to powers of two.
E8M0 = torch.float8_e8m0fnu


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Frobenius-norm relative error.

    Elementwise tolerances are a poor fit for a GEMM: a single output can sit
    near zero through cancellation, where any absolute difference looks
    enormous in relative terms while contributing nothing to the result.
    """
    return (actual.float() - expected.float()).norm().item() / expected.float().norm().item()


def rounding_error(expected: torch.Tensor) -> float:
    """What a perfectly correct kernel still loses by writing its result as bf16."""
    return relative_error(expected.bfloat16(), expected)


#: Accumulation-order slack for a float32 output. Measured at ~2e-8.
ACCUMULATION_TOLERANCE = 1e-6

#: How much worse than correct rounding a bf16 output is allowed to be.
ROUNDING_SLACK = 1.05


def fp8_operands(m: int, n: int, k: int, block_size: int):
    """A quantized activation and weight, plus the float32 pair they decode to."""
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE) * 0.1

    a, a_s = act_quant(x, block_size, "ue8m0", E8M0)
    b, b_s = fp8_weight_quant(w, block_size, "ue8m0", E8M0)

    a_ref = act_quant(x, block_size, "ue8m0", E8M0, dequant=True).float()
    b_ref = fp8_weight_quant(w, block_size, "ue8m0", E8M0, dequant=True).float()
    return (a, a_s, b, b_s), (a_ref, b_ref)


@requires_sm90
@pytest.mark.parametrize("block_size", [32, 128])
def test_fp8_gemm_matches_its_dequantized_operands(block_size):
    (a, a_s, b, b_s), (a_ref, b_ref) = fp8_operands(64, 256, 512, block_size)
    expected = a_ref @ b_ref.T

    exact = fp8_gemm(a, a_s, b, b_s, scale_dtype=E8M0, block_size=block_size, out_dtype=torch.float32)
    rounded = fp8_gemm(a, a_s, b, b_s, scale_dtype=E8M0, block_size=block_size)

    assert relative_error(exact, expected) < ACCUMULATION_TOLERANCE
    assert relative_error(rounded, expected) <= rounding_error(expected) * ROUNDING_SLACK


@requires_sm90
@pytest.mark.parametrize("act_block_size", [32, 128])
def test_fp4_gemm_matches_its_dequantized_operands(act_block_size):
    """The weight's block is fixed at 32 while the activation's is a parameter.

    That asymmetry is the checkpoint's, not a test convenience: fp4 has one
    mantissa bit, so it needs a scale every 32 values to stay usable, whereas
    the activation is quantized at run time and can afford coarser blocks.
    """
    x = torch.randn(64, 512, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(256, 512, dtype=torch.bfloat16, device=DEVICE) * 0.1

    a, a_s = act_quant(x, act_block_size, "ue8m0", E8M0)
    b, b_s = fp4_act_quant(w, 32)

    a_ref = act_quant(x, act_block_size, "ue8m0", E8M0, dequant=True).float()
    b_ref = fp4_act_quant(w, 32, dequant=True).float()
    expected = a_ref @ b_ref.T

    exact = fp4_gemm(a, a_s, b, b_s, scale_dtype=E8M0, act_block_size=act_block_size, out_dtype=torch.float32)
    rounded = fp4_gemm(a, a_s, b, b_s, scale_dtype=E8M0, act_block_size=act_block_size)

    assert relative_error(exact, expected) < ACCUMULATION_TOLERANCE
    assert relative_error(rounded, expected) <= rounding_error(expected) * ROUNDING_SLACK


@requires_sm90
def test_fp4_is_half_the_bytes_of_fp8():
    """The packing is the whole reason fp4 is worth a separate kernel.

    ``float4_e2m1fn_x2`` stores two values per byte along K, so the tensor is
    declared with half the trailing dimension. A kernel written against the
    logical width would silently read past the row.
    """
    w = torch.randn(256, 512, dtype=torch.bfloat16, device=DEVICE)

    b, b_s = fp4_act_quant(w, 32)

    assert b.dtype == torch.float4_e2m1fn_x2
    assert b.shape == (256, 256)
    assert b_s.shape == (256, 16)
    assert b.untyped_storage().nbytes() == 256 * 256


@requires_sm90
def test_the_output_dtype_does_not_follow_the_global_default():
    """A deliberate divergence from the reference implementation.

    Upstream allocates the output with ``torch.get_default_dtype()`` while
    compiling the kernel for bf16. That only agrees because the reference sets
    a global default of bf16 at import. VeOmni sets no global default, so under
    ``torch.set_default_dtype(torch.float32)`` the upstream form would hand back
    a float32 buffer holding bf16 bytes -- garbage, silently. The dtype is a
    parameter here, and the kernel is compiled to match it.
    """
    (a, a_s, b, b_s), (a_ref, b_ref) = fp8_operands(64, 256, 512, 128)
    expected = a_ref @ b_ref.T

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        default_shaped = fp8_gemm(a, a_s, b, b_s, scale_dtype=E8M0, block_size=128)
        asked_for_float32 = fp8_gemm(a, a_s, b, b_s, scale_dtype=E8M0, block_size=128, out_dtype=torch.float32)
    finally:
        torch.set_default_dtype(previous)

    assert default_shaped.dtype == torch.bfloat16
    assert relative_error(default_shaped, expected) <= rounding_error(expected) * ROUNDING_SLACK
    assert asked_for_float32.dtype == torch.float32
    assert relative_error(asked_for_float32, expected) < ACCUMULATION_TOLERANCE


@requires_sm90
def test_the_unquantized_original_would_be_a_useless_reference():
    """Why these tests compare against dequantized operands, not the bf16 source.

    fp4 keeps one mantissa bit, and it costs about 12% of the weight. A test
    written against the original ``x @ w.T`` would need a tolerance roughly two
    orders of magnitude wider than the kernel's own error, at which point it
    stops testing the kernel and starts testing the quantizer.
    """
    w = torch.randn(256, 512, dtype=torch.bfloat16, device=DEVICE)

    loss = relative_error(fp4_act_quant(w, 32, dequant=True), w)

    assert 5e-2 < loss < 2e-1


@requires_sm90
def test_an_unsupported_output_dtype_is_refused():
    (a, a_s, b, b_s), _ = fp8_operands(64, 256, 512, 128)

    with pytest.raises(ValueError, match="out_dtype"):
        fp8_gemm(a, a_s, b, b_s, scale_dtype=E8M0, block_size=128, out_dtype=torch.float16)
