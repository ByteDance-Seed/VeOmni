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
from torch import Tensor

from ....qat import fp8_fake_quant_act, fp8_fake_quant_weight


def wrapper(x: Tensor, weight: Tensor, bias: Tensor, *, block_size: int = 128) -> Tensor:
    """Run ``F.linear`` with blockwise FP8 fake-quantized activation and weight.

    An empty ``bias`` tensor denotes an unused bias. ``block_size`` is shared by
    the ``1 x block_size`` activation blocks and the square weight tiles, so a
    caller reproduces a deployment recipe by choosing it here.
    """
    if x.numel() == 0:
        return F.linear(x, weight, None if bias.numel() == 0 else bias)
    x_q = fp8_fake_quant_act(x, block_size=block_size)
    w_q = fp8_fake_quant_weight(weight, block_size=block_size)
    return F.linear(x_q, w_q, None if bias.numel() == 0 else bias)
