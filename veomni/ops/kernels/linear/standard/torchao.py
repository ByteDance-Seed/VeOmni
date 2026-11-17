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

"""Standard dense linear NVFP4 QAT implementation.

Both GEMM operands are fake-quantized to NVFP4 (16-wide E4M3 block scales with
an FP32 per-tensor scale) through torchao, and the multiply itself runs in the
model dtype -- the same emulation shape as the TileLang ``qat`` row next to it.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ....qat.torchao_fp4 import torchao_fake_quant
from .._module import resolve_projection


def wrapper(
    linear: nn.Module,
    x: Tensor,
    *,
    dtype: str = "nvfp4",
    block_size: int | None = None,
) -> Tensor:
    """Run ``linear`` with both GEMM operands fake-quantized through torchao.

    ``dtype`` picks the format: ``nvfp4`` (16-wide groups, an FP32 per-tensor scale)
    or ``mxfp4`` (32-wide groups, an E8M0 power-of-two scale). ``block_size``
    defaults to that format's group width, which is the only width it defines.
    """
    # A quantizing row always gets a weight: ``resolve_projection`` raises for a
    # wrapped projection instead of handing back a shortcut.
    weight, shortcut = resolve_projection("torchao", linear, x)
    assert shortcut is None
    bias = getattr(linear, "bias", None)
    quantize = torchao_fake_quant(dtype)
    return F.linear(quantize(x, block_size), quantize(weight, block_size), bias)
