# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Which activation a SiLU-fused expert row can compute itself."""

from __future__ import annotations

import torch


__all__ = ["activation_backward", "activation_is_not_silu", "delegate_target", "is_silu"]


def activation_is_not_silu(activation: str) -> bool:
    """Whether *activation* has to leave a SiLU-fused kernel.

    ``veomni.ops.activation.normalize_activation`` raises for a name outside its
    closed set, but such a config is still runnable: the ``eager`` row falls back to
    the model's own ``act_fn``, which is what the model ran before the activation
    became a call argument. So this compares spellings rather than validating them --
    the fallback is exactly what handles the names that validation would reject.
    """
    name = activation.value if hasattr(activation, "value") else activation
    return name not in ("silu", "swish")


def delegate_target(activation: str) -> str:
    """Which row serves a SiLU-fused kernel that cannot run *activation*.

    ``eager`` is the reference implementation and handles every activation name,
    including ones outside ``veomni.ops.activation``. But it indexes the expert
    weights it is handed by *global* expert id, so under expert parallelism -- where
    each rank holds a shard -- it would read past the shard or pick the wrong expert.
    ``triton`` owns the EP dispatch, so it is the only row that can serve a
    non-Swish activation there.

    Returns:
        ``"triton"`` when expert parallelism is on, ``"eager"`` otherwise.
    """
    from .....distributed.parallel_state import get_parallel_state

    return "triton" if get_parallel_state().ep_enabled else "eager"


def is_silu(activation: str) -> bool:
    """Whether *activation* is the one a SiLU-fused kernel computes.

    A plain spelling comparison, not ``normalize_activation``: that raises for a name
    outside ``veomni.ops.activation``, and such a config is still runnable through the
    model's own ``act_fn``. Asking "is this SiLU?" must answer ``False`` for those, not
    raise.
    """
    name = activation.value if hasattr(activation, "value") else activation
    return name in ("silu", "swish")


def activation_backward(grad_output, hidden, activation: str, act_fn=None):
    """``grad_output * activation'(hidden)``, for a row's hand-written backward.

    The four names ``veomni.ops.activation`` serves go through its ATen backward ops.
    Anything else is differentiated through the caller's own ``act_fn``, which is what
    the model ran before the activation became a call argument -- so a ``relu`` or
    ``mish`` config keeps working instead of failing at its first backward.

    Args:
        grad_output: Upstream gradient.
        hidden: The activation's input, in whatever dtype forward used.
        activation: The configured name.
        act_fn: The model's activation module, required when *activation* is outside
            the served set.
    """
    from ....activation import apply_activation_backward, supported_activations, supports

    if supports(activation):
        return apply_activation_backward(grad_output, hidden, activation)
    if act_fn is None:
        raise ValueError(
            f"No activation module for {activation!r}. This row evaluates "
            f"{sorted(supported_activations())}, and the caller passed no ``act_fn`` "
            f"to differentiate anything else."
        )
    # A fresh leaf under ``enable_grad``: this runs inside a custom Function's
    # backward, where grad mode is already off.
    with torch.enable_grad():
        leaf = hidden.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(act_fn(leaf), leaf, grad_output)
    return grad
