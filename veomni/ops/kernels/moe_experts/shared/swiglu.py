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

"""SwiGLU pre-activation clamp and gated activations shared by the fused MoE paths."""

import torch


def apply_swiglu_clamp(fc1_1_output, fc1_2_output, swiglu_limit):
    """gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation.

    Returns ``(fc1_1_clamped, fc1_2_clamped, mask_fc1_1, mask_fc1_2)``.
    When ``swiglu_limit`` is ``None`` this is a no-op and masks are ``None``;
    callers may skip the corresponding mask multiplications in backward.

    Semantics mirror ``torch.clamp`` autograd: gradients vanish where the
    pre-clamp value falls outside the bound. ``gate`` is upper-bounded only
    (``max=limit``) so the negative tail still feeds SiLU; ``up`` is
    symmetrically clamped to ``[-limit, +limit]``.
    """
    if swiglu_limit is None:
        return fc1_1_output, fc1_2_output, None, None
    mask_fc1_1 = fc1_1_output <= swiglu_limit
    mask_fc1_2 = (fc1_2_output >= -swiglu_limit) & (fc1_2_output <= swiglu_limit)
    fc1_1_output = fc1_1_output.clamp(max=swiglu_limit)
    fc1_2_output = fc1_2_output.clamp(min=-swiglu_limit, max=swiglu_limit)
    return fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply SwiGLU-OAI after any model-specific input clamping.

    Evaluated as ``(up + 1) * (gate * sigmoid(gate * alpha))``, the op order of
    the MiniMax / GPT-OSS eager experts. Every step rounds in bf16, so any other
    association drifts from eager by an ULP on a large share of elements.
    """
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


def swiglu_oai_backward(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return analytical gradients for the unclamped SwiGLU-OAI inputs."""
    sigmoid = torch.sigmoid(alpha * gate)
    glu = gate * sigmoid
    grad_gate = grad_output * (up + 1.0) * (sigmoid + alpha * gate * sigmoid * (1.0 - sigmoid))
    grad_up = grad_output * glu
    return grad_gate, grad_up


def swiglu_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_alpha: float | None,
    *,
    return_gate_activation: bool = False,
):
    """Fused-MoE gated activation: standard SwiGLU, or SwiGLU-OAI when ``swiglu_alpha`` is set.

    With ``return_gate_activation`` also returns ``silu(gate)`` (``None`` for
    SwiGLU-OAI) so a caller recomputing the activation in backward can hand it to
    :func:`swiglu_activation_backward` instead of evaluating SiLU twice.
    """
    if swiglu_alpha is None:
        gate_activation = torch.ops.aten.silu(gate)
        output = gate_activation * up
    else:
        gate_activation = None
        output = swiglu_oai(gate, up, swiglu_alpha)
    return (output, gate_activation) if return_gate_activation else output


def swiglu_activation_backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_alpha: float | None,
    gate_activation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of :func:`swiglu_activation` w.r.t. its (post-clamp) ``gate`` and ``up``.

    ``gate_activation`` is an already computed ``silu(gate)`` for the standard
    path; it is recomputed when omitted.
    """
    if swiglu_alpha is None:
        if gate_activation is None:
            gate_activation = torch.ops.aten.silu(gate)
        grad_gate = torch.ops.aten.silu_backward(grad_output * up, gate)
        grad_up = gate_activation * grad_output
        return grad_gate, grad_up
    return swiglu_oai_backward(grad_output, gate, up, swiglu_alpha)
