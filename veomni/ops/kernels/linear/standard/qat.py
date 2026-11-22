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

"""Standard dense linear blockwise QAT implementation.

Both GEMM operands are fake-quantized with the DeepSeek-V4 TileLang quantizers
and the multiply itself runs in the model dtype, which is what an emulated
low-precision deployment GEMM computes: the rounded values are read back and
accumulated in high precision.

``dtype`` is FP8-only (``"fp8"``), using ``1 x block_size`` activation blocks
and square ``block_size x block_size`` weight tiles with optional power-of-two
scales.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ....qat import fp8_fake_quant_act, fp8_fake_quant_weight
from .._module import resolve_projection


#: Formats this row can round to, as its ``dtype`` argument.
SUPPORTED_DTYPES = ("fp8",)


def wrapper(linear: nn.Module, x: Tensor, *, dtype: str = "fp8", block_size: int | None = None) -> Tensor:
    """Run ``linear`` with blockwise fake-quantized activation and weight.

    ``dtype`` must be ``"fp8"``. ``block_size`` (default 128) sets both operands'
    granularity: ``1 x block_size`` activation blocks and square
    ``block_size x block_size`` weight tiles. The deployment GEMM ties them together
    -- it indexes a weight's scale by the same block counter as the activation's --
    so one number describes its recipe, not two.
    """
    # A quantizing row always gets a weight: ``resolve_projection`` raises for a
    # wrapped projection instead of handing back a shortcut.
    weight, shortcut = resolve_projection("qat", linear, x)
    assert shortcut is None
    bias = getattr(linear, "bias", None)
    if x.numel() == 0:
        return F.linear(x, weight, bias)
    if dtype == "fp4":
        # FP4 rounds the weight only: the checkpoint's scales are per weight group,
        # and the activation layout of the FP4 expert kernels is not one a dense
        # projection consumes. A dense FP4 GEMM is not what this row is for.
        raise ValueError(
            "linear qat with dtype='fp4' is not implemented: the FP4 recipe in this tree is the routed-expert"
            " one (32-wide input groups, weight-only scales). Use dtype='fp8' for dense projections."
        )
    if dtype != "fp8":
        raise ValueError(f"linear qat does not support dtype={dtype!r}; it supports {SUPPORTED_DTYPES}")
    width = 128 if block_size is None else block_size
    x_q = fp8_fake_quant_act(x, block_size=width)
    w_q = fp8_fake_quant_weight(weight, block_size=width)
    return F.linear(x_q, w_q, bias)
