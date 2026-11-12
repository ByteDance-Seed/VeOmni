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

* :func:`normalize_activation` / :func:`supports` let a row validate a selection
  without knowing the config's vocabulary.
* :func:`apply_activation` / :func:`apply_activation_backward` evaluate one.

**Every name delegates to the transformer's own implementation**, looked up by the
name the config used. That is not redundancy: the names are not interchangeable.
``gelu_new`` resolves to ``NewGELUActivation``, which stages ``torch.pow(x, 3.0)``,
while ``gelu_pytorch_tanh`` resolves to ``PytorchGELUTanh``, which calls
``F.gelu(approximate="tanh")`` and writes ``x * x * x`` internally. In bf16 the
intermediates round differently -- max ~1.6e-2 on unit-scale inputs -- so mapping
them onto one formula would change a model's outputs and gradients against the
function it trained with.

A Triton elementwise kernel was considered and rejected. The activation sits
between GEMMs whose operands are already quantized through TileLang, so it is a
rounding error in the profile, and the ``eager`` row has to stay a readable
reference: a reference implementation that dispatches to a kernel on the one
device where it is hardest to check is not a reference.

SiLU keeps its ATen spelling so its values and gradients match the fused kernels
bit for bit.
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "apply_activation",
    "apply_activation_backward",
    "normalize_activation",
    "supported_activations",
    "supports",
]


#: Names this module evaluates, exactly as a config spells them. Membership is
#: checked against ``transformers.activations.ACT2FN`` at import time so a
#: transformers upgrade that renames or drops one fails here, not mid-training.
ACTIVATIONS: tuple[str, ...] = (
    "silu",
    "swish",
    "gelu",
    "gelu_new",
    "gelu_pytorch_tanh",
)


@lru_cache(maxsize=None)
def _resolver(name: str):
    """The transformers implementation for *name*, or ``None`` if it has none.

    ``ACT2FN`` maps most names to a *class* (``NewGELUActivation``,
    ``PytorchGELUTanh``) and a few to a bare function. Instantiating the classes
    once here keeps callers free of that distinction, and the cache is what makes
    repeated evaluation during training cheap.
    """
    from transformers.activations import ACT2FN

    if name == "swish":
        # HF aliases this to SiLU, and naming it explicitly keeps the lookup from
        # depending on whether that alias survives a transformers upgrade.
        name = "silu"
    implementation = ACT2FN.get(name)
    if implementation is None:
        return None
    if inspect.isclass(implementation):
        return implementation()
    return implementation


def supported_activations() -> frozenset[str]:
    """Every config-level activation name this module can evaluate."""
    return frozenset(name for name in ACTIVATIONS if _resolver(name) is not None)


def supports(activation: str) -> bool:
    """Whether this module can evaluate *activation* on this host."""
    return activation in supported_activations()


def normalize_activation(activation: str) -> str:
    """Return the canonical spelling of *activation*.

    Raises:
        ValueError: if the name has no implementation. Refusing is the point: the
            caller has a config asking for a function the row cannot compute, and
            substituting SiLU would train a different model than the one
            described.
    """
    name = activation.value if hasattr(activation, "value") else activation
    if name == "swish":
        name = "silu"
    if _resolver(name) is None:
        raise ValueError(
            f"Unsupported MLP activation {name!r}. The gated-MLP rows implement "
            f"{sorted(supported_activations())}. Select a row that evaluates this activation, or "
            f"leave the model on its `eager` projection fallback."
        )
    return name


def apply_activation(hidden: Tensor, activation: str) -> Tensor:
    """Evaluate ``activation(hidden)`` through transformers' own implementation."""
    name = normalize_activation(activation)
    if name == "silu":
        # ATen, not ``ACT2FN["silu"]``: identical values, and the fused rows and
        # HF's autograd both use this spelling, so gradients stay bit-comparable.
        return F.silu(hidden)
    return _resolver(name)(hidden)


def apply_activation_backward(grad_output: Tensor, hidden: Tensor, activation: str) -> Tensor:
    """Return ``grad_output * activation'(hidden)``.

    Autograd computes the derivative rather than this module restating it. Callers
    are inside a custom ``autograd.Function``, but they are not inside a graph
    that would be re-entered: ``hidden`` is a saved tensor, and this runs once per
    backward. Restating three derivative formulas was where a missing ``up``
    factor survived a bit-exact forward.
    """
    name = normalize_activation(activation)
    if name == "silu":
        return torch.ops.aten.silu_backward(grad_output, hidden)
    with torch.enable_grad():
        leaf = hidden.detach().requires_grad_(True)
        value = _resolver(name)(leaf)
        (grad,) = torch.autograd.grad(value, leaf, grad_output)
    return grad
