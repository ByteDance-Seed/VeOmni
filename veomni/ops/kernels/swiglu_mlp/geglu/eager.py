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
from torch import Tensor

from ....registry import SavedState
from ..standard.eager import empty_output, linear, linear_backward


_SQRT_2_OVER_PI = 0.7978845608028654
_GELU_TANH_COEFF = 0.044715


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
    """Return ``(grad_gate, grad_up)`` for ``gelu_pytorch_tanh(gate) * up``."""
    gelu_gate = F.gelu(gate, approximate="tanh")
    grad_up = grad_hidden * gelu_gate
    gate_f = gate.float()
    gate_sq = gate_f * gate_f
    tanh_arg = _SQRT_2_OVER_PI * (gate_f + _GELU_TANH_COEFF * gate_f * gate_sq)
    tanh_res = torch.tanh(tanh_arg)
    dgelu = 0.5 * (1.0 + tanh_res) + 0.5 * gate_f * (1.0 - tanh_res * tanh_res) * (
        _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_TANH_COEFF * gate_sq)
    )
    grad_gate = grad_hidden * up * dgelu.to(dtype=gate.dtype)
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
