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

"""standard SwiGLU MLP eager math (gate / up / silu-mul / down)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ....registry import SavedState


@dataclass(frozen=True)
class _Meta:
    """Empty path, clamp, and which biases were real tensors."""

    empty: bool
    swiglu_limit: float | None
    has_gate_bias: bool
    has_up_bias: bool
    has_down_bias: bool


def optional_bias(bias: Tensor) -> Tensor | None:
    """Return ``None`` when *bias* is the empty unused-layout sentinel."""
    return None if bias.numel() == 0 else bias


def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """``F.linear`` that treats an empty bias as unused."""
    return F.linear(x, weight, optional_bias(bias))


def clamp_gate_up(gate: Tensor, up: Tensor, swiglu_limit: float | None) -> tuple[Tensor, Tensor]:
    """DeepSeek-V4 clamp: gate has an upper cap, up is two-sided."""
    if swiglu_limit is None:
        return gate, up
    return gate.clamp(max=swiglu_limit), up.clamp(min=-swiglu_limit, max=swiglu_limit)


def silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """``silu(gate) * up``."""
    return F.silu(gate) * up


def silu_mul_backward(grad_hidden: Tensor, gate: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(grad_gate, grad_up)`` for ``silu(gate) * up``."""
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    grad_up = grad_hidden * silu_gate
    grad_gate = grad_hidden * up * (silu_gate * (1 - sig) + sig)
    return grad_gate, grad_up


def unclamp_gate(grad_clamped: Tensor, gate: Tensor, swiglu_limit: float | None) -> Tensor:
    """Zero the gate grad where the upper clamp was active."""
    if swiglu_limit is None:
        return grad_clamped
    return grad_clamped * (gate <= swiglu_limit).to(dtype=grad_clamped.dtype)


def unclamp_up(grad_clamped: Tensor, up: Tensor, swiglu_limit: float | None) -> Tensor:
    """Zero the up grad where the two-sided clamp was active."""
    if swiglu_limit is None:
        return grad_clamped
    return grad_clamped * ((up >= -swiglu_limit) & (up <= swiglu_limit)).to(dtype=grad_clamped.dtype)


def linear_backward(
    grad_output: Tensor,
    inp: Tensor,
    weight: Tensor,
    *,
    has_bias: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return ``(grad_input, grad_weight, grad_bias)`` for ``F.linear``."""
    grad_2d = grad_output.reshape(-1, weight.shape[0])
    inp_2d = inp.reshape(-1, weight.shape[1])
    grad_input = (grad_2d @ weight).reshape_as(inp)
    grad_weight = grad_2d.transpose(0, 1) @ inp_2d
    grad_bias = grad_2d.sum(dim=0) if has_bias else None
    return grad_input, grad_weight, grad_bias


def mlp_hidden(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    *,
    swiglu_limit: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project, optional clamp, then ``silu * up``.

    When *swiglu_limit* is set, clamp and the silu-mul run in fp32, then the
    hidden state is cast back to ``x.dtype`` before the down projection,
    matching official DeepSeek-V4 eager shared-expert math.
    """
    gate = linear(x, gate_w, gate_b)
    up = linear(x, up_w, up_b)
    if swiglu_limit is not None:
        gate = gate.float()
        up = up.float()
    gate_c, up_c = clamp_gate_up(gate, up, swiglu_limit)
    hidden = silu_mul(gate_c, up_c)
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)
    return hidden, gate, up


def empty_output(x: Tensor, down_w: Tensor) -> Tensor:
    """Output with the down-projection width and the leading shape of *x*."""
    return x.new_zeros(*x.shape[:-1], down_w.shape[0])


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
    """Full SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    Empty biases are unused. ``swiglu_limit`` is the DeepSeek-V4 clamp; ``None``
    skips it.
    """
    meta = _Meta(
        x.numel() == 0,
        swiglu_limit,
        gate_b.numel() > 0,
        up_b.numel() > 0,
        down_b.numel() > 0,
    )
    if meta.empty:
        return empty_output(x, down_w), SavedState((x, gate_w, gate_b, up_w, up_b, down_w, down_b), meta)

    hidden, gate, up = mlp_hidden(x, gate_w, gate_b, up_w, up_b, swiglu_limit=swiglu_limit)
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
        grad_output, hidden_out, down_w, has_bias=meta.has_down_bias
    )
    if meta.swiglu_limit is not None:
        gate_c, up_c = clamp_gate_up(gate, up, meta.swiglu_limit)
        grad_hidden = grad_hidden.to(dtype=gate_c.dtype)
        grad_gate_c, grad_up_c = silu_mul_backward(grad_hidden, gate_c, up_c)
        grad_gate = unclamp_gate(grad_gate_c, gate, meta.swiglu_limit).to(dtype=x.dtype)
        grad_up = unclamp_up(grad_up_c, up, meta.swiglu_limit).to(dtype=x.dtype)
    else:
        grad_gate, grad_up = silu_mul_backward(grad_hidden, gate, up)

    grad_x_gate, grad_gate_w, grad_gate_b = linear_backward(grad_gate, x, gate_w, has_bias=meta.has_gate_bias)
    grad_x_up, grad_up_w, grad_up_b = linear_backward(grad_up, x, up_w, has_bias=meta.has_up_bias)
    return (
        grad_x_gate + grad_x_up,
        grad_gate_w,
        grad_gate_b,
        grad_up_w,
        grad_up_b,
        grad_down_w,
        grad_down_b,
    )
