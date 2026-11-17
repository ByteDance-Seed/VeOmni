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
from torch import Tensor, nn

from ....activation import apply_activation, supports


@dataclass(frozen=True)
class _Meta:
    """Empty path, clamp, and which biases were real tensors."""

    empty: bool
    swiglu_limit: float | None
    projection_dtype: torch.dtype
    has_gate_bias: bool
    has_up_bias: bool
    has_down_bias: bool
    activation: str = "silu"


def optional_bias(bias: Tensor) -> Tensor | None:
    """Return ``None`` when *bias* is the empty unused-layout sentinel."""
    return None if bias.numel() == 0 else bias


def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """``F.linear`` that treats an empty bias as unused."""
    return F.linear(x, weight, optional_bias(bias))


def clamp_gate_up(gate: Tensor, up: Tensor, swiglu_limit: float | None) -> tuple[Tensor, Tensor]:
    """DeepSeek-V4 clamp, in whatever dtype the caller passes: gate capped, up two-sided."""
    if swiglu_limit is None:
        return gate, up
    return gate.clamp(max=swiglu_limit), up.clamp(min=-swiglu_limit, max=swiglu_limit)


def clamp_gate_up_in_fp32(gate: Tensor, up: Tensor, swiglu_limit: float | None) -> tuple[Tensor, Tensor]:
    """The same clamp, widened to fp32 first -- official DeepSeek-V4 shared-expert math.

    One helper for both of ``eager``'s paths (by weight, and through the projection
    modules) so a bf16 model clamps identically whichever path runs. It is kept
    separate from :func:`clamp_gate_up` because the quantizing rows deliberately
    clamp in the model dtype, matching the deployment kernel they emulate.
    """
    if swiglu_limit is None:
        return gate, up
    return clamp_gate_up(gate.float(), up.float(), swiglu_limit)


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
    compute_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return Linear grads using the dtype selected during its forward."""
    grad_2d = grad_output.reshape(-1, weight.shape[0]).to(dtype=compute_dtype)
    inp_2d = inp.reshape(-1, weight.shape[1]).to(dtype=compute_dtype)
    weight_compute = weight.to(dtype=compute_dtype)
    grad_input = (grad_2d @ weight_compute).reshape_as(inp).to(dtype=inp.dtype)
    grad_weight = (grad_2d.transpose(0, 1) @ inp_2d).to(dtype=weight.dtype)
    grad_bias = grad_2d.sum(dim=0).to(dtype=weight.dtype) if has_bias else None
    return grad_input, grad_weight, grad_bias


def mlp_hidden(
    x: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    *,
    swiglu_limit: float | None,
    activation: str = "silu",
) -> tuple[Tensor, Tensor, Tensor, torch.dtype]:
    """Project, optional clamp, then ``activation(gate) * up``.

    When *swiglu_limit* is set, clamp and the silu-mul run in fp32, then the
    hidden state is cast back to ``x.dtype`` before the down projection,
    matching official DeepSeek-V4 eager shared-expert math.
    """
    gate = linear(x, gate_w, gate_b)
    up = linear(x, up_w, up_b)
    projection_dtype = gate.dtype
    gate_c, up_c = clamp_gate_up_in_fp32(gate, up, swiglu_limit)
    hidden = apply_activation(gate_c, activation) * up_c
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)
    return hidden, gate, up, projection_dtype


def empty_output(x: Tensor, down_w: Tensor) -> Tensor:
    """Output with the down-projection width and the leading shape of *x*."""
    return x.new_zeros(*x.shape[:-1], down_w.shape[0])


def wrapper(
    mlp: nn.Module,
    x: Tensor,
    *,
    swiglu_limit: float | None = None,
    activation: str = "silu",
) -> Tensor:
    """Run the gated MLP as plain PyTorch, letting autograd build the backward.

    This is the row entry point. It deliberately does **not** go through
    ``forward`` / ``backward`` above: those exist for ``liger_kernel`` to reuse the
    clamp and projection helpers, and ``register_op`` turns a raw pair into one
    ``autograd.Function`` wrapping the whole MLP -- which two other mechanisms
    cannot coexist with:

    * **Activation offload** runs the backward inside ``saved_tensors_hooks``
      (``veomni/distributed/offloading.py``). ``torch.func.vjp`` refuses to run
      there ("don't yet support saved tensor hooks"), and a hand-written derivative
      cannot reproduce the bf16 rounding of the model's own autograd chain, so a
      non-SiLU activation under offload would either raise or train different
      gradients.
    * **``torch.compile(fullgraph=True)``** rejects a ``requires_grad_`` flag flip,
      which is the other way to get a derivative inside a custom Function.

    Plain PyTorch sidesteps both: there is no Function to enter, so the hooks see
    the individual ops and Dynamo traces them. Verified bit-identical to the raw
    pair for ``silu`` and ``gelu``, with and without the clamp, in bf16 and fp32 --
    so switching the registration does not change any trained numerics.

    ``mlp`` may also be a projection the row cannot read (a LoRA wrapper, an
    ``nn.Linear`` subclass) or an activation nothing here evaluates; then the
    projections run as modules, which is the model's own path. See
    :func:`.._module.render`.
    """
    from .._module import render

    rendered = render(mlp, impl="eager", activation=activation)
    if rendered.projections is not None:
        gate_proj, up_proj, down_proj = rendered.projections
        gate, up = clamp_gate_up_in_fp32(gate_proj(x), up_proj(x), swiglu_limit)
        if supports(activation):
            hidden = apply_activation(gate, activation) * up
        else:
            # An activation ``veomni.ops.activation`` does not evaluate. The block
            # still has its own ``act_fn``, which is what the model ran before this
            # family existed and is exact for any activation -- so the reference row
            # keeps working for every config that worked before.
            hidden = mlp.act_fn(gate) * up
        return down_proj(hidden.to(x.dtype))

    # No empty-input shortcut here: ``F.linear`` on a zero-token batch returns zeros
    # with a live graph, and a detached zero would cut the block out of the backward
    # pass for exactly the ranks that hold no tokens. The fused rows keep their own
    # shortcut, because kernels that misbehave on an empty grid need one.
    gate_w, gate_b, up_w, up_b, down_w, down_b = rendered.weights
    hidden, _, _, _ = mlp_hidden(x, gate_w, gate_b, up_w, up_b, swiglu_limit=swiglu_limit, activation=activation)
    return linear(hidden, down_w, down_b)
