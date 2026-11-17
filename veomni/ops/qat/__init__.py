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

"""Fake-quantization primitives for quantization-aware training.

These are functional helpers: model code calls them directly because which
tensor gets quantized at which granularity is a property of the model's
quantization recipe rather than a selectable op implementation.

The recipes implemented here are DeepSeek-V4's: block-wise FP8 everywhere except
the routed experts of a V4-Flash checkpoint, whose weights are FP4. NVIDIA uses
the TileLang quantizers and AMD gfx942/gfx950 uses their Triton FP8 port.

These are *primitives*, not a selection mechanism. Which tensor is rounded, at
what block width, is chosen by the op row that calls them -- ``linear`` /
``swiglu_mlp`` / ``moe_experts`` / ``kv_round`` each have their own
implementation name for it. Quantizing calls load the backend lazily, so
importing this package on CPU or NPU stays free.

:mod:`veomni.ops.qat.torchao_nvfp4` is separate: it builds the NVFP4 round trip
on torchao instead of TileLang, has no SM90 requirement, and is imported
directly rather than re-exported here.
"""

from . import fp8_blockwise
from ._hardware import require_tilelang_sm90
from .fp4_blockwise import (
    FP4_BLOCK_SIZE,
    fp4_fake_quant_weight,
)
from .fp8_blockwise import (
    DEFAULT_SCALE_FMT,
    fp8_fake_quant_act,
    fp8_fake_quant_act_prefix,
    fp8_fake_quant_stacked_weight,
    fp8_fake_quant_weight,
)


def act_quant(*args, **kwargs):
    """Run this GPU's FP8 activation quantizer lazily."""
    return fp8_blockwise.act_quant(*args, **kwargs)


def fp4_act_quant(*args, **kwargs):
    """Run the optional TileLang FP4 quantizer lazily."""
    require_tilelang_sm90()
    from .quant import fp4_act_quant as impl

    return impl(*args, **kwargs)


def fp8_weight_quant(*args, **kwargs):
    """Run this GPU's FP8 weight quantizer lazily."""
    return fp8_blockwise.fp8_weight_quant(*args, **kwargs)


__all__ = [
    "DEFAULT_SCALE_FMT",
    "FP4_BLOCK_SIZE",
    "act_quant",
    "fp4_act_quant",
    "fp4_fake_quant_weight",
    "fp8_fake_quant_act",
    "fp8_fake_quant_act_prefix",
    "fp8_fake_quant_stacked_weight",
    "fp8_fake_quant_weight",
    "fp8_weight_quant",
]
