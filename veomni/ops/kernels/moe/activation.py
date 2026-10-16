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

import torch


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


def swiglu_activation(gate: torch.Tensor, up: torch.Tensor, swiglu_alpha: float | None) -> torch.Tensor:
    """Fused-MoE gated activation: standard SwiGLU, or SwiGLU-OAI when ``swiglu_alpha`` is set."""
    if swiglu_alpha is None:
        return torch.ops.aten.silu(gate) * up
    return swiglu_oai(gate, up, swiglu_alpha)


def swiglu_activation_backward(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, swiglu_alpha: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of :func:`swiglu_activation` w.r.t. its (post-clamp) ``gate`` and ``up``."""
    if swiglu_alpha is None:
        grad_gate = torch.ops.aten.silu_backward(grad_output * up, gate)
        grad_up = torch.ops.aten.silu(gate) * grad_output
        return grad_gate, grad_up
    return swiglu_oai_backward(grad_output, gate, up, swiglu_alpha)
