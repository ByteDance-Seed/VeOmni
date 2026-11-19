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

from torch import Tensor, nn

from ....registry import SavedState, _make_autograd_fn
from . import eager as _eager


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
    activation: str = "silu",
) -> tuple[Tensor, SavedState]:
    """Same MLP as eager; ``silu(gate) * up`` uses Liger.

    No activation check here: :func:`.._module.render` has already established that
    this row serves the activation before reaching this, and it is the single place
    that decides. A guard here would be unreachable -- and being unreachable is why
    the ``standard`` variant's non-SiLU behaviour is tested through ``render``.
    """
    # No empty-input branch here: :func:`wrapper` routes ``x.numel() == 0`` to the
    # ``eager`` row before this is reached, and ``_eager.forward``'s own empty path
    # returns a detached zero, which would cut the block out of the backward pass.
    gate = _eager.linear(x, gate_w, gate_b)
    up = _eager.linear(x, up_w, up_b)
    projection_dtype = gate.dtype
    meta = _eager._Meta(
        empty=False,
        swiglu_limit=swiglu_limit,
        projection_dtype=projection_dtype,
        has_gate_bias=gate_b.numel() > 0,
        has_up_bias=up_b.numel() > 0,
        has_down_bias=down_b.numel() > 0,
    )
    if swiglu_limit is not None:
        gate = gate.float()
        up = up.float()
    gate_c, up_c = _eager.clamp_gate_up(gate, up, swiglu_limit)

    from liger_kernel.ops.swiglu import swiglu_forward

    saved_gate, saved_up, hidden = swiglu_forward(gate_c.contiguous(), up_c.contiguous())
    if hidden.dtype != x.dtype:
        hidden = hidden.to(dtype=x.dtype)
    output = _eager.linear(hidden, down_w, down_b)
    return output, SavedState(
        (x, gate_w, gate_b, up_w, up_b, down_w, down_b, gate, up, hidden, saved_gate, saved_up),
        meta,
    )


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
    """Down linear, Liger silu-mul, then the two input linears."""
    meta = saved.metadata
    assert isinstance(meta, _eager._Meta)
    x, gate_w, gate_b, up_w, up_b, down_w, down_b, gate, up, hidden, saved_gate, saved_up = saved.tensors
    grad_hidden, grad_down_w, grad_down_b = _eager.linear_backward(
        grad_output,
        hidden,
        down_w,
        has_bias=meta.has_down_bias,
        compute_dtype=grad_output.dtype,
    )

    from liger_kernel.ops.swiglu import swiglu_backward

    if grad_hidden.dtype != saved_gate.dtype:
        grad_hidden = grad_hidden.to(dtype=saved_gate.dtype)
    grad_gate_c, grad_up_c = swiglu_backward(saved_gate, saved_up, grad_hidden.contiguous())
    if meta.swiglu_limit is not None:
        grad_gate = _eager.unclamp_gate(grad_gate_c.to(dtype=gate.dtype), gate, meta.swiglu_limit).to(
            dtype=meta.projection_dtype
        )
        grad_up = _eager.unclamp_up(grad_up_c.to(dtype=up.dtype), up, meta.swiglu_limit).to(
            dtype=meta.projection_dtype
        )
    else:
        grad_gate = grad_gate_c
        grad_up = grad_up_c

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


#: The differentiable entry point for this row's raw math.
#:
#: ``forward`` is not one: it runs ``swiglu_forward``, a bare Triton kernel, and
#: hand-written ``F.linear`` calls, none of which build a graph. ``_make_autograd_fn``
#: is what turns the raw pair into an ``autograd.Function``, and it is also what
#: ``register_op`` would have built had this row been registered as a pair.
_autograd = _make_autograd_fn(forward, backward)


def wrapper(
    mlp: nn.Module,
    x: Tensor,
    *,
    swiglu_limit: float | None = None,
    activation: str = "silu",
) -> Tensor:
    """Row entry point: take the MLP module and run it through Liger.

    Liger's ``swiglu_forward`` / ``swiglu_backward`` implement SiLU only, so a model
    that asks for another activation -- or whose projections cannot be read by
    weight -- is served by its own projection modules instead of by this row. That
    decision is :func:`.._module.render`'s, so a model's MLP forward stays a single
    handle call.
    """
    from .._module import render

    rendered = render(mlp, impl="liger_kernel", activation=activation)
    if rendered.projections is not None or x.numel() == 0:
        # An empty grid is not what Liger is for, and the raw pair's empty path
        # returns a detached zero -- which would cut the block out of the backward
        # pass. Plain PyTorch is exact on zero elements and keeps the graph.
        return _eager.wrapper(mlp, x, swiglu_limit=swiglu_limit, activation=activation)

    # ``_autograd``, not ``forward``: the kernels in ``forward`` build no graph, so
    # calling it directly would return a tensor whose gradient never reaches
    # ``gate_proj`` / ``up_proj`` -- which is exactly what it used to do.
    gate_w, gate_b, up_w, up_b, down_w, down_b = rendered.weights
    return _autograd(
        x,
        gate_w,
        gate_b,
        up_w,
        up_b,
        down_w,
        down_b,
        swiglu_limit=swiglu_limit,
        activation=activation,
    )
