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


def build(in_features, out_features, quantized):
    from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Linear

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
    """``<param>_scale`` is the converter's single rule; a mismatch is an unexpected key."""
    state = build(64, 32, quantized=True).state_dict()

    assert state["weight"].dtype == torch.float8_e4m3fn
    assert state["weight_scale"].dtype == torch.float8_e8m0fnu
    assert state["weight_scale"].shape == (1, 2)  # 32/32 rows by 64/32 columns


def test_quantized_linear_refuses_eager_op():
    """Pre-quantized FP8 weights fail fast when run through the eager training op."""
    layer = build(64, 32, quantized=True)
    with pytest.raises(ValueError, match="holds pre-quantized weights"):
        layer(torch.randn(4, 64, dtype=torch.bfloat16))


def test_an_fp4_weight_is_refused_for_a_linear():
    """fp4 is only ever a stacked expert slab, which is not an nn.Linear."""
    with pytest.raises(ValueError, match="not a quantized storage dtype"):
        build(64, 32, quantized=False).__class__(64, 32, quant_dtype=torch.float4_e2m1fn_x2)
