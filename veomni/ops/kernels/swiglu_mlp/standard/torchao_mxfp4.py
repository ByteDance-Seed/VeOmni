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

"""standard SwiGLU MLP MXFP4 QAT implementation.

Mirrors ``standard/eager.py`` exactly except that every projection consumes
fake-quantized operands. The clamp and ``silu * up`` stay in high precision: they
run on the dequantized projection outputs, not on stored values, so no
deployment kernel rounds them.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ....qat.torchao_mxfp4 import MXFP4_BLOCK_SIZE, local_shard, mxfp4_fake_quant
from . import eager as _eager


def _linear(x: Tensor, weight: Tensor, bias: Tensor, block_size: int) -> Tensor:
    """``F.linear`` over MXFP4 fake-quantized operands, empty bias meaning unused."""
    return torch.nn.functional.linear(
        mxfp4_fake_quant(x, block_size),
        mxfp4_fake_quant(weight, block_size),
        None if bias.numel() == 0 else local_shard(bias),
    )


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
    block_size: int = MXFP4_BLOCK_SIZE,
) -> Tensor:
    """Run SwiGLU with MXFP4 fake-quantized projections and the optional DeepSeek-V4 clamp.

    Empty bias tensors denote unused biases. ``block_size`` is the MX group width.
    A zero-token ``x`` needs no special case: every operand stays attached to the
    graph, so the output is empty and backward yields zero gradients on the
    frozen weights, which is what the ``eager`` and ``liger_kernel`` rows do.
    """
    gate = _linear(x, gate_w, gate_b, block_size)
    up = _linear(x, up_w, up_b, block_size)
    if swiglu_limit is not None:
        gate = gate.float()
        up = up.float()
    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
    hidden = _eager.silu_mul(gate_c, up_c)
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)

    return _linear(hidden, down_w, down_b, block_size)
