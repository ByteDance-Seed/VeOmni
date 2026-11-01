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

"""geglu MLP Liger adapter (linears + fused gelu-tanh-mul + down)."""

from __future__ import annotations

from torch import Tensor

from ....registry import SavedState
from . import eager as _eager


def forward(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
) -> tuple[Tensor, SavedState]:
    """Same MLP as eager; ``gelu_tanh(gate) * up`` uses Liger when nonempty."""
    if x.numel() == 0:
        return _eager.forward(x, gate_w, gate_b, up_w, up_b, down_w, down_b)

    gate = _eager.linear(x, gate_w, gate_b)
    up = _eager.linear(x, up_w, up_b)
    meta = _eager._Meta(
        empty=False,
        projection_dtype=gate.dtype,
        has_gate_bias=gate_b.numel() > 0,
        has_up_bias=up_b.numel() > 0,
        has_down_bias=down_b.numel() > 0,
    )

    from liger_kernel.ops.geglu import geglu_forward

    saved_gate, saved_up, hidden = geglu_forward(gate.contiguous(), up.contiguous())
    output = _eager.linear(hidden, down_w, down_b)
    return output, SavedState(
        (x, gate_w, gate_b, up_w, up_b, down_w, down_b, hidden, saved_gate, saved_up),
        meta,
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Down linear, Liger gelu-tanh-mul, then the two input linears."""
    meta = saved.metadata
    assert isinstance(meta, _eager._Meta)
    if meta.empty:
        return _eager.backward(grad_output, saved)

    x, gate_w, gate_b, up_w, up_b, down_w, down_b, hidden, saved_gate, saved_up = saved.tensors
    grad_hidden, grad_down_w, grad_down_b = _eager.linear_backward(
        grad_output,
        hidden,
        down_w,
        has_bias=meta.has_down_bias,
        compute_dtype=grad_output.dtype,
    )

    from liger_kernel.ops.geglu import geglu_backward

    if grad_hidden.dtype != saved_gate.dtype:
        grad_hidden = grad_hidden.to(dtype=saved_gate.dtype)
    grad_gate, grad_up = geglu_backward(saved_gate, saved_up, grad_hidden.contiguous())

    grad_x_gate, grad_gate_w, grad_gate_b = _eager.linear_backward(
        grad_gate,
        x,
        gate_w,
        has_bias=meta.has_gate_bias,
        compute_dtype=meta.projection_dtype,
    )
    grad_x_up, grad_up_w, grad_up_b = _eager.linear_backward(
        grad_up,
        x,
        up_w,
        has_bias=meta.has_up_bias,
        compute_dtype=meta.projection_dtype,
    )
    return (
        grad_x_gate + grad_x_up,
        grad_gate_w,
        grad_gate_b,
        grad_up_w,
        grad_up_b,
        grad_down_w,
        grad_down_b,
    )
