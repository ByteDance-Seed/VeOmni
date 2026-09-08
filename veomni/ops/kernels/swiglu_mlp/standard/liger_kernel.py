# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""standard SwiGLU MLP Liger adapter (linears + fused silu-mul + down)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


@dataclass(frozen=True)
class _Meta:
    """Empty / eager-fallback flag, clamp, and which biases were real."""

    empty: bool
    swiglu_limit: float | None
    has_gate_bias: bool
    has_up_bias: bool
    has_down_bias: bool


def forward(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
    *,
    swiglu_limit: float | None = None,
) -> tuple[Tensor, SavedState]:
    """Same MLP as eager; ``silu(gate) * up`` uses Liger when the tensors are nonempty."""
    meta = _Meta(
        x.numel() == 0,
        swiglu_limit,
        gate_b.numel() > 0,
        up_b.numel() > 0,
        down_b.numel() > 0,
    )
    if meta.empty:
        output, saved = _eager.forward(x, gate_w, gate_b, up_w, up_b, down_w, down_b, swiglu_limit=swiglu_limit)
        return output, SavedState(saved.tensors, meta)

    gate = _eager.linear(x, gate_w, gate_b)
    up = _eager.linear(x, up_w, up_b)
    if swiglu_limit is not None:
        gate = gate.float()
        up = up.float()
    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)
    if gate_c.dtype != x.dtype:
        gate_c = gate_c.to(dtype=x.dtype)
        up_c = up_c.to(dtype=x.dtype)

    from liger_kernel.ops.swiglu import swiglu_forward

    saved_gate, saved_up, hidden = swiglu_forward(gate_c.contiguous(), up_c.contiguous())
    output = _eager.linear(hidden, down_w, down_b)
    return output, SavedState(
        (x, gate_w, gate_b, up_w, up_b, down_w, down_b, gate, up, hidden, saved_gate, saved_up),
        meta,
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Down linear, Liger silu-mul, then the two input linears."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        return _eager.backward(grad_output, SavedState(saved.tensors, _eager._Meta(*meta)))

    x, gate_w, gate_b, up_w, up_b, down_w, down_b, gate, up, hidden, saved_gate, saved_up = saved.tensors
    grad_hidden, grad_down_w, grad_down_b = _eager.linear_backward(
        grad_output, hidden, down_w, has_bias=meta.has_down_bias
    )

    from liger_kernel.ops.swiglu import swiglu_backward

    grad_gate_c, grad_up_c = swiglu_backward(saved_gate, saved_up, grad_hidden.contiguous())
    if meta.swiglu_limit is not None:
        grad_gate = _eager.unclamp_gate(grad_gate_c.to(dtype=gate.dtype), gate, meta.swiglu_limit).to(dtype=x.dtype)
        grad_up = _eager.unclamp_up(grad_up_c.to(dtype=up.dtype), up, meta.swiglu_limit).to(dtype=x.dtype)
    else:
        grad_gate = grad_gate_c
        grad_up = grad_up_c

    grad_x_gate, grad_gate_w, grad_gate_b = _eager.linear_backward(grad_gate, x, gate_w, has_bias=meta.has_gate_bias)
    grad_x_up, grad_up_w, grad_up_b = _eager.linear_backward(grad_up, x, up_w, has_bias=meta.has_up_bias)
    return (
        grad_x_gate + grad_x_up,
        grad_gate_w,
        grad_gate_b,
        grad_up_w,
        grad_up_b,
        grad_down_w,
        grad_down_b,
    )
