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

"""Activation selection shared by the gated-MLP rows.

Both ``swiglu_mlp`` and ``moe_experts`` compute ``act_fn(gate) * up``. Which
``act_fn`` is a property of the *model*, not of the row: a config says
``hidden_act: gelu`` and expects its MLP to use GELU whether or not the row
quantizes. Hard-coding SiLU inside a row made that impossible, so a model with
another activation fell back to its projection modules -- and a quantizing row
that fell back trained at full precision while the config claimed otherwise.

This module is deliberately pure PyTorch, with no kernel of its own:

* :func:`normalize_activation` maps the names a config uses onto the names a row
  supports, so a row can validate a selection without knowing the config's
  vocabulary.
* :func:`apply_activation` / :func:`apply_activation_backward` evaluate one.

A Triton elementwise kernel was considered and rejected. The activation sits
between GEMMs whose operands are already quantized through TileLang, so it is a
rounding error in the profile, and the ``eager`` row has to stay a readable
reference: a reference implementation that dispatches to a kernel on the one
device where it is hardest to check is not a reference. SiLU keeps its ATen
spelling because the fused kernels and HF's autograd both use it, and matching
their rounding bit-for-bit is what lets ``qat`` reuse ``eager``'s gradient.
"""

from __future__ import annotations

import math
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "ACTIVATION_ALIASES",
    "ActivationType",
    "apply_activation",
    "apply_activation_backward",
    "normalize_activation",
    "supported_activations",
]


class ActivationType(str, Enum):
    """Activations this module can evaluate, named after their HF spelling."""

    GELU = "gelu"
    GELU_PYTORCH_TANH = "gelu_pytorch_tanh"
    SILU = "silu"


#: Model-config spellings mapped onto :class:`ActivationType` values. The left
#: side is what ``transformers.activations.ACT2FN`` accepts; a name missing here
#: is refused rather than silently rounded to SiLU.
ACTIVATION_ALIASES: dict[str, str] = {
    # ``gelu`` is the exact erf form. ``gelu_new`` and ``gelu_tanh`` are HF's
    # other names for the tanh approximation, and they are *not* numerically the
    # same function -- mapping them together would change a model's math.
    "gelu": ActivationType.GELU.value,
    "gelu_new": ActivationType.GELU_PYTORCH_TANH.value,
    "gelu_pytorch_tanh": ActivationType.GELU_PYTORCH_TANH.value,
    "gelu_tanh": ActivationType.GELU_PYTORCH_TANH.value,
    "silu": ActivationType.SILU.value,
    # HF's own alias for SiLU (``ACT2FN["swish"] is ACT2FN["silu"]``). Omitting
    # it would refuse a config that is bit-for-bit the row's default.
    "swish": ActivationType.SILU.value,
}


def supported_activations() -> frozenset[str]:
    """Every config-level activation name this module can evaluate."""
    return frozenset(ACTIVATION_ALIASES)


def normalize_activation(activation: str | ActivationType) -> str:
    """Map a model-config activation name onto a canonical name.

    Raises:
        ValueError: if the name has no implementation here. Refusing is the
            point: the caller has a config asking for a function the row cannot
            compute, and substituting SiLU would train a different model than
            the one described.
    """
    name = activation.value if isinstance(activation, ActivationType) else activation
    try:
        return ACTIVATION_ALIASES[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported MLP activation {name!r}. The gated-MLP rows implement "
            f"{sorted(ACTIVATION_ALIASES)}. Select a row that supports this activation, or use the "
            f"model's `eager` fallback."
        ) from error


def apply_activation(hidden: Tensor, activation: str | ActivationType) -> Tensor:
    """Evaluate ``activation(hidden)``.

    SiLU goes through ATen so its values and gradients match the fused kernels
    and HF's autograd bit for bit.
    """
    name = normalize_activation(activation)
    if name == ActivationType.SILU.value:
        return F.silu(hidden)
    if name == ActivationType.GELU.value:
        return F.gelu(hidden)
    return F.gelu(hidden, approximate="tanh")


def apply_activation_backward(
    grad_output: Tensor,
    hidden: Tensor,
    activation: str | ActivationType,
) -> Tensor:
    """Return ``grad_output * activation'(hidden)``.

    The derivative is written out rather than obtained from autograd: callers are
    inside a custom ``autograd.Function``, where nesting another graph would cost
    a full backward pass for one elementwise term.
    """
    name = normalize_activation(activation)
    if name == ActivationType.SILU.value:
        return torch.ops.aten.silu_backward(grad_output, hidden)

    x = hidden.float()
    if name == ActivationType.GELU.value:
        derivative = _gelu_grad(x)
    else:
        derivative = _gelu_pytorch_tanh_grad(x)
    return (grad_output.float() * derivative).to(hidden.dtype)


def _gelu_grad(x: Tensor) -> Tensor:
    """Derivative of the exact (erf) GELU."""
    cdf = 0.5 * (1.0 + torch.erf(x * math.sqrt(0.5)))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return cdf + x * pdf


def _gelu_pytorch_tanh_grad(x: Tensor) -> Tensor:
    """Derivative of the tanh-approximated GELU, as ``F.gelu(approximate="tanh")`` computes it."""
    inner = 0.79788456 * (x + 0.044715 * x * x * x)
    tanh_inner = torch.tanh(inner)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner * tanh_inner) * 0.79788456 * (
        1.0 + 3 * 0.044715 * x * x
    )
