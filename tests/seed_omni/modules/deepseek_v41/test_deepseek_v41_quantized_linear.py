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

"""The linear layer that reads the released checkpoint's fp8 weights."""

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.dequant import dequant_fp8_blockwise
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()

requires_sm90 = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or torch.version.hip is not None or get_gpu_compute_capability() < 90,
    reason="the quantized path runs on TileLang kernels needing an SM90 or later NVIDIA GPU",
)


def build(in_features, out_features, quantized):
    from veomni.models.seed_omni.modules.deepseek_v41.llm.quantized import DeepseekV41Linear

    return DeepseekV41Linear(
        in_features,
        out_features,
        quant_dtype=torch.float8_e4m3fn if quantized else None,
    )


def test_an_unquantized_layer_is_still_exactly_nn_linear():
    """The bf16 checkpoint and the training path must see no change at all."""
    layer = build(64, 32, quantized=False)
    x = torch.randn(4, 64)

    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, layer.weight), rtol=0, atol=0)
    assert not layer.is_quantized
    assert "weight_scale" not in layer.state_dict()


def test_a_quantized_layer_declares_the_names_the_converter_writes():
    """``<param>_scale`` is the converter's single rule; a mismatch is an unexpected key.

    Checked as names rather than by loading, because this is the one thing the
    two halves have to agree on and neither can see the other.
    """
    state = build(64, 32, quantized=True).state_dict()

    assert state["weight"].dtype == torch.float8_e4m3fn
    assert state["weight_scale"].dtype == torch.float8_e8m0fnu
    assert state["weight_scale"].shape == (1, 2)  # 32/32 rows by 64/32 columns


@requires_sm90
def test_a_quantized_layer_matches_its_own_dequantized_weight():
    """Isolates the layer from the quantizer by giving both sides the same weight.

    The remaining difference is the activation, which the layer quantizes to fp8
    before the GEMM and the reference does not -- about 2.6e-2. Rather than
    allow that as a tolerance, it is measured: running the same product with a
    quantize-dequantized activation and an untouched weight gives the floor the
    layer cannot beat, and the layer is held to it. A loose absolute bound here
    would pass a layer that had lost an order of magnitude somewhere else.
    """
    from veomni.models.seed_omni.modules.deepseek_v41.llm.quantized import act_quant

    torch.manual_seed(0)
    layer = build(256, 128, quantized=True).to(DEVICE)
    reference_weight = torch.randn(128, 256, dtype=torch.bfloat16, device=DEVICE) * 0.1
    quantized_weight, scale = act_quant(reference_weight, 32, "ue8m0", torch.float8_e8m0fnu)
    # act_quant scales per row; the layer's scale is per tile. One row block of
    # 32 covering the whole width is the case where the two coincide.
    with torch.no_grad():
        layer.weight.copy_(quantized_weight)
        layer.weight_scale.copy_(scale[::32, :])

    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE)
    weight = dequant_fp8_blockwise(layer.weight, layer.weight_scale, torch.float32)
    expected = x.float() @ weight.T

    def relative_error(actual):
        return ((actual.float() - expected).norm() / expected.norm()).item()

    floor = relative_error(act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, dequant=True).float() @ weight.T)

    assert relative_error(layer(x)) <= floor * 1.05


@requires_sm90
def test_the_quantized_and_plain_paths_agree_on_shape():
    quantized = build(256, 128, quantized=True).to(DEVICE)
    plain = build(256, 128, quantized=False).to(DEVICE).to(torch.bfloat16)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE)

    assert quantized(x).shape == plain(x).shape == (8, 128)


def test_an_fp4_weight_is_refused_for_a_linear():
    """fp4 is only ever a stacked expert slab, which is not an nn.Linear.

    Allocating one here would produce a layer whose scale is shaped for a
    stack, and the mismatch would not surface until a GEMM read past a row.
    """
    with pytest.raises(ValueError, match="not a quantized storage dtype"):
        build(64, 32, quantized=False).__class__(64, 32, quant_dtype=torch.float4_e2m1fn_x2)
