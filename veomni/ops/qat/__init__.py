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

The recipes implemented so far are DeepSeek-V4's: block-wise FP8 everywhere
except the routed experts of a V4-Flash checkpoint, whose weights are FP4. Both
reuse the V4 TileLang quantizers in :mod:`veomni.ops.qat.quant`, so training
stays bit-aligned with inference by construction; AMD gfx942/gfx950 GPUs run the
bit-exact Triton FP8 port in :mod:`veomni.ops.qat.triton_quant`. FP8 calls
require one of those GPUs and FP4 calls an NVIDIA SM90 or later GPU; both load
the backend lazily, so importing this package on CPU or NPU stays free.
``qat_linear(..., enabled=False)`` bypasses quantization and its hardware and
dependency checks.
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
    qat_linear,
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
    "qat_linear",
]
