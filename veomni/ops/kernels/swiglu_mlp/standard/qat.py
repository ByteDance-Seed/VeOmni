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

"""standard gated-MLP blockwise-FP8 QAT implementation.

Mirrors ``standard/eager.py`` except that every projection consumes fake-quantized
operands. The clamp and ``act_fn(gate) * up`` run in the model dtype on the
dequantized projection outputs, not in the FP32 space ``eager`` uses: a deployment
fused kernel consumes BF16 projection outputs and clamps them there, so widening
to FP32 would train a slightly different function than the one that will run.

The activation is a call argument, not part of the row. Quantization touches only
the GEMM operands -- ``act_fn`` is an elementwise step between them -- so
``down(act_fn(gate(x)) * up(x))`` fake-quantizes identically whatever the
activation is. Making it a call argument is what lets a model with ``gelu`` use
this row instead of falling back to its projection modules unquantized.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ....activation import apply_activation
from ....qat import fp8_fake_quant_act, fp8_fake_quant_weight
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
    block_size: int = 128,
    activation: str = "silu",
) -> Tensor:
    """Run a gated MLP with blockwise FP8 fake-quantized projections.

    Empty bias tensors denote unused biases. ``activation`` is the config's
    ``hidden_act``; it defaults to SiLU because that is what every published
    checkpoint of this shape uses, and the string is validated by
    :func:`~veomni.ops.activation.apply_activation`.
    """
    x_gate_q = fp8_fake_quant_act(x, block_size=block_size) if x.numel() > 0 else x
    gate = F.linear(x_gate_q, fp8_fake_quant_weight(gate_w, block_size=block_size), _eager.optional_bias(gate_b))
    up = F.linear(x_gate_q, fp8_fake_quant_weight(up_w, block_size=block_size), _eager.optional_bias(up_b))

    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
    hidden = apply_activation(gate_c, activation) * up_c
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)

    hidden_q = fp8_fake_quant_act(hidden, block_size=block_size) if hidden.numel() > 0 else hidden
    return F.linear(hidden_q, fp8_fake_quant_weight(down_w, block_size=block_size), _eager.optional_bias(down_b))
