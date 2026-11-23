# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Activation handling shared by the ``moe_experts`` rows.

A fused row computes SiLU inside its grouped GEMM, so any other activation has to be
evaluated outside it. The name the model configured is the whole input needed: every
model in this repository builds its ``act_fn`` as ``ACT2FN[config.hidden_act]``, so
``veomni.ops.activation`` can resolve the same callable from the name alone.
"""

from __future__ import annotations

import torch


__all__ = ["activation_backward", "is_silu"]


def is_silu(activation: str) -> bool:
    """Whether *activation* is the one a SiLU-fused kernel computes.

    A plain spelling comparison rather than ``normalize_activation``: that raises for a
    name outside the served set, and such a config is still runnable through the
    reference row. Asking "is this SiLU?" must answer ``False`` for those, not raise.
    """
    name = activation.value if hasattr(activation, "value") else activation
    return name in ("silu", "swish")


def activation_backward(grad_output, hidden, activation: str):
    """``grad_output * activation'(hidden)``, for a row's hand-written backward.

    The names ``veomni.ops.activation`` serves go through its ATen backward ops --
    bit-exact against ``ACT2FN``'s autograd. Anything else is differentiated through
    the same callable the model configured, so a ``relu`` or ``mish`` expert keeps
    working instead of failing at its first backward.
    """
    from ....activation import apply_activation_backward, resolve, supports

    if supports(activation):
        return apply_activation_backward(grad_output, hidden, activation)
    # A fresh leaf under ``enable_grad``: this runs inside a custom Function's
    # backward, where grad mode is already off.
    act_fn = resolve(activation)
    with torch.enable_grad():
        leaf = hidden.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(act_fn(leaf), leaf, grad_output)
    return grad
