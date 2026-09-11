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

These are library ops: model code calls them directly instead of going through
`KERNEL_REGISTRY` / `OpSlot`, because which tensor gets quantized at which
granularity is a property of the model's quantization recipe rather than a
kernel the user picks.

The recipes implemented so far are DeepSeek-V4's: block-wise FP8 everywhere
except the routed experts of a V4-Flash checkpoint, whose weights are FP4. Both
reuse the V4 TileLang quantizers (`veomni.ops.kernels.deepseek_v4`), so training
stays bit-aligned with inference by construction. They are SM90-only and loaded
lazily, so importing this package on CPU or NPU stays free.
"""

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


__all__ = [
    "DEFAULT_SCALE_FMT",
    "FP4_BLOCK_SIZE",
    "fp4_fake_quant_weight",
    "fp8_fake_quant_act",
    "fp8_fake_quant_act_prefix",
    "fp8_fake_quant_stacked_weight",
    "fp8_fake_quant_weight",
    "qat_linear",
]
