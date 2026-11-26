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

"""geglu MLP eager math (gate / up / gelu-tanh-mul / down)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ....registry import SavedState
from ..standard.eager import empty_output, linear, linear_backward


@dataclass(frozen=True)
class _Meta:
    """Empty path and which biases were real tensors."""

    empty: bool
    projection_dtype: torch.dtype
    has_gate_bias: bool
    has_up_bias: bool
    has_down_bias: bool


def gelu_tanh_mul(gate: Tensor, up: Tensor) -> Tensor:
    """``gelu_pytorch_tanh(gate) * up``."""
    return F.gelu(gate, approximate="tanh") * up


def gelu_tanh_mul_backward(grad_hidden: Tensor, gate: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    """Return gate/up gradients with the same operation order as PyTorch autograd."""
    gelu_gate = F.gelu(gate, approximate="tanh")
    grad_up = grad_hidden * gelu_gate
    # Match MulBackward followed by GeluBackward, including low-precision
    # rounding of the upstream gradient before the activation derivative.
    grad_gate = torch.ops.aten.gelu_backward(grad_hidden * up, gate, approximate="tanh")
    return grad_gate, grad_up


def mlp_hidden(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor, torch.dtype]:
    """Project, then ``gelu_tanh * up``."""
    gate = linear(x, gate_w, gate_b)
    up = linear(x, up_w, up_b)
    return gelu_tanh_mul(gate, up), gate, up, gate.dtype


def forward(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
) -> tuple[Tensor, SavedState]:
    """Full GeGLU MLP: ``down(gelu_pytorch_tanh(gate(x)) * up(x))``.

    Empty biases are unused. The activation is the tanh approximation used by
    Gemma, not exact GELU.
    """
    if x.numel() == 0:
        meta = _Meta(True, x.dtype, gate_b.numel() > 0, up_b.numel() > 0, down_b.numel() > 0)
        return empty_output(x, down_w), SavedState((x, gate_w, gate_b, up_w, up_b, down_w, down_b), meta)

    hidden, gate, up, projection_dtype = mlp_hidden(x, gate_w, gate_b, up_w, up_b)
    meta = _Meta(False, projection_dtype, gate_b.numel() > 0, up_b.numel() > 0, down_b.numel() > 0)
    output = linear(hidden, down_w, down_b)
    return output, SavedState((x, gate_w, gate_b, up_w, up_b, down_w, down_b, gate, up, hidden), meta)


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Return grads for ``x`` and the six weight / bias tensors."""
    meta = saved.metadata
    assert isinstance(meta, _Meta)
    if meta.empty:
        x, gate_w, gate_b, up_w, up_b, down_w, down_b = saved.tensors
        return (
            torch.zeros_like(x),
            torch.zeros_like(gate_w),
            None if not meta.has_gate_bias else torch.zeros_like(gate_b),
            torch.zeros_like(up_w),
            None if not meta.has_up_bias else torch.zeros_like(up_b),
            torch.zeros_like(down_w),
            None if not meta.has_down_bias else torch.zeros_like(down_b),
        )

    x, gate_w, gate_b, up_w, up_b, down_w, down_b, gate, up, hidden_out = saved.tensors
    grad_hidden, grad_down_w, grad_down_b = linear_backward(
        grad_output,
        hidden_out,
        down_w,
        has_bias=meta.has_down_bias,
        compute_dtype=grad_output.dtype,
    )
    grad_hidden = grad_hidden.to(dtype=meta.projection_dtype)
    grad_gate, grad_up = gelu_tanh_mul_backward(grad_hidden, gate, up)

    grad_x_gate, grad_gate_w, grad_gate_b = linear_backward(
        grad_gate,
        x,
        gate_w,
        has_bias=meta.has_gate_bias,
        compute_dtype=meta.projection_dtype,
    )
    grad_x_up, grad_up_w, grad_up_b = linear_backward(
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


#: The two config spellings this variant computes. Both resolve to the same
#: tanh-approximate GELU, which is what ``gelu_tanh_mul`` evaluates.
TANH_GELU = frozenset({"gelu_pytorch_tanh", "gelu_python_tanh"})


def wrapper(mlp: nn.Module, x: Tensor, *, activation: str = "gelu_pytorch_tanh") -> Tensor:
    """Row entry point: take the GeGLU MLP module and run it in the model dtype.

    The variant is fixed to tanh-approximate GELU, so *activation* is not a way to
    run something else -- it is the model's ``hidden_act``, passed so the row can
    tell when it is the wrong row. Gemma3 configures ``gelu_pytorch_tanh`` today;
    a config that asks for another activation still has to compute that activation,
    so the row runs the projection modules through the model's own ``act_fn``, which
    is exact for any of them. That check used to sit in the model's forward, as an
    ``if`` around the handle call.

    A projection the row cannot read by weight -- a LoRA wrapper, an ``nn.Linear``
    subclass -- takes the same module path, as does an empty batch.
    """
    from .._module import plain_projections

    gate_proj, up_proj, down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if activation not in TANH_GELU:
        # The config asks for another activation than this variant computes, so the
        # row cannot be the one that runs it. The block's own ``act_fn`` is exact for
        # any of them, and it is the expression the model used before this family.
        return down_proj(mlp.act_fn(gate_proj(x)) * up_proj(x))

    if not plain_projections(mlp) or x.numel() == 0:
        # A projection the row cannot read by weight runs its own ``forward``, and a
        # zero-token batch does too: ``forward``'s empty shortcut returns a detached
        # zero, which would cut the block out of the backward pass for exactly the
        # ranks that hold no tokens. The modules are exact on zero elements and keep
        # the graph.
        return down_proj(gelu_tanh_mul(gate_proj(x), up_proj(x)))

    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    empty = gate.weight.new_empty(0)
    output, _ = forward(
        x,
        gate.weight,
        gate.bias if gate.bias is not None else empty,
        up.weight,
        up.bias if up.bias is not None else empty,
        down.weight,
        down.bias if down.bias is not None else empty,
    )
    return output
