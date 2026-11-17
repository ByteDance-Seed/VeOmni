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

"""Standard SwiGLU MLP NVFP4 QAT implementation.

Mirrors ``standard/qat.py`` except that every projection consumes NVFP4
fake-quantized operands (16-wide E4M3 block scales with an FP32 per-tensor
scale) from torchao. As there, the clamp and ``silu * up`` run in the model
dtype on the dequantized projection outputs: a deployment fused kernel consumes
BF16 projection outputs and clamps them there.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ....qat.torchao_fp4 import torchao_fake_quant
from . import eager as _eager


def wrapper(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
    *,
    swiglu_limit: float | None = None,
    dtype: str = "nvfp4",
    block_size: int | None = None,
) -> Tensor:
    """Run SwiGLU with torchao fake-quantized projections and the optional clamp.

    ``dtype`` picks the format (``nvfp4`` or ``mxfp4``); ``block_size`` defaults to
    that format's group width. Empty bias tensors denote unused biases.
    """
    quantize = torchao_fake_quant(dtype)
    x_q = quantize(x, block_size)
    gate = F.linear(x_q, quantize(gate_w, block_size), _eager.optional_bias(gate_b))
    up = F.linear(x_q, quantize(up_w, block_size), _eager.optional_bias(up_b))

    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
    hidden = _eager.silu_mul(gate_c, up_c)
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)

    hidden_q = quantize(hidden, block_size)
    return F.linear(hidden_q, quantize(down_w, block_size), _eager.optional_bias(down_b))
