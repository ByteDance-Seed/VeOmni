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

"""Standard dense linear blockwise-FP8 QAT implementation.

Both GEMM operands are fake-quantized with the DeepSeek-V4 TileLang quantizers
and the multiply itself runs in the model dtype, which is what an emulated FP8
deployment GEMM computes: the rounded values are read back and accumulated in
high precision.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ....qat import fp8_fake_quant_act, fp8_fake_quant_weight
from .._module import resolve_projection


def wrapper(linear: nn.Module, x: Tensor, *, block_size: int = 128) -> Tensor:
    """Run ``linear`` with blockwise FP8 fake-quantized activation and weight.

    ``block_size`` sets both operands' granularity: ``1 x block_size`` activation
    blocks and square ``block_size x block_size`` weight tiles. The deployment GEMM
    ties them together -- it indexes a weight's scale by the same block counter as
    the activation's -- so one number describes its recipe, not two.
    """
    # A quantizing row always gets a weight: ``resolve_projection`` raises for a
    # wrapped projection instead of handing back a shortcut.
    weight, shortcut = resolve_projection("qat", linear, x)
    assert shortcut is None
    bias = getattr(linear, "bias", None)
    if x.numel() == 0:
        return F.linear(x, weight, bias)
    x_q = fp8_fake_quant_act(x, block_size=block_size)
    w_q = fp8_fake_quant_weight(weight, block_size=block_size)
    return F.linear(x_q, w_q, bias)
