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
from torch import Tensor, nn

from ....activation import apply_activation
from ....qat.torchao_nvfp4 import NVFP4_BLOCK_SIZE, nvfp4_fake_quant
from . import eager as _eager


def wrapper(
    mlp: nn.Module,
    x: Tensor,
    *,
    swiglu_limit: float | None = None,
    block_size: int = NVFP4_BLOCK_SIZE,
    activation: str = "silu",
) -> Tensor:
    """Run a gated MLP with NVFP4 fake-quantized projections and the optional clamp.

    ``mlp`` is the model's MLP block; see the ``qat`` row for why this one reads
    weights and therefore refuses a wrapped projection rather than quantizing
    something the projection does not actually compute.
    """
    from .._module import render

    gate_w, gate_b, up_w, up_b, down_w, down_b = render(mlp, impl="torchao_nvfp4", activation=activation).weights
    x_q = nvfp4_fake_quant(x, block_size)
    gate = F.linear(x_q, nvfp4_fake_quant(gate_w, block_size), _eager.optional_bias(gate_b))
    up = F.linear(x_q, nvfp4_fake_quant(up_w, block_size), _eager.optional_bias(up_b))

    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
    hidden = apply_activation(gate_c, activation) * up_c
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)

    hidden_q = nvfp4_fake_quant(hidden, block_size)
    return F.linear(hidden_q, nvfp4_fake_quant(down_w, block_size), _eager.optional_bias(down_b))
