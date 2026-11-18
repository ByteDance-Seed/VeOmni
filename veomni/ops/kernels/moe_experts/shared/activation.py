# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Which activation a SiLU-fused expert row can compute itself."""

from __future__ import annotations


__all__ = ["activation_is_not_silu"]


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
