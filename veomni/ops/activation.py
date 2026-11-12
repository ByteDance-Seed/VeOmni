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


#: ``ACT2FN`` entries this module will not evaluate, with the reason. Everything
#: else in ``ACT2FN`` is served as-is, so a transformers release that adds an
#: activation needs no change here -- only a genuinely incompatible one does.
_DISALLOWED: dict[str, str] = {
    "prelu": "carries a learnable `num_parameters` slope, so an inline instance would train "
    "a different function from the model's own",
    "xielu": "carries learnable `alpha_p` / `alpha_n` parameters, same problem as `prelu`",
    "laplace": "normalizes along the last dimension rather than acting elementwise, so "
    "`activation(gate) * up` would not mean what the row assumes",
    "gelu_10": "its forward takes `min` / `max` bounds in addition to the input, so it is not a "
    "one-argument activation",
}


@lru_cache(maxsize=None)
def _resolver(name: str):
    """The transformers implementation for *name*, or ``None`` if it has none.

    ``ACT2FN`` mixes three shapes: a *class* (``NewGELUActivation``), an already
    instantiated ``nn.Module`` (``ClippedGELUActivation()``), and occasionally a
    bare function. Normalizing to something callable here keeps callers free of
    that, and the cache is what makes repeated evaluation during training cheap.
    """
    from transformers.activations import ACT2FN

    if name == "swish":
        # HF aliases this to SiLU, and naming it explicitly keeps the lookup from
        # depending on whether that alias survives a transformers upgrade.
        name = "silu"
    if name not in ACT2FN:
        return None
    # Subscript, not ``.get``: ``ACT2FN`` is a ``ClassInstantier`` whose ``__getitem__``
    # turns a stored ``(cls, kwargs)`` pair into an instance. ``.get`` returns that
    # raw pair, which is not callable.
    implementation = ACT2FN[name]
    if inspect.isclass(implementation):
        implementation = implementation()
    return implementation if callable(implementation) else None


def supported_activations() -> frozenset[str]:
    """Every config-level activation name this module can evaluate.

    Derived from ``ACT2FN`` so the set tracks the installed transformers rather
    than a list that drifts from it. ``swish`` is added explicitly because HF
    aliases it to ``silu`` and the alias has not always been present.
    """
    from transformers.activations import ACT2FN

    return (frozenset(ACT2FN) - frozenset(_DISALLOWED)) | {"swish"}


def supports(activation: str) -> bool:
    """Whether this module can evaluate *activation* on this host."""
    return activation in supported_activations()


def normalize_activation(activation: str) -> str:
    """Return the canonical spelling of *activation*.

    Raises:
        ValueError: if the name is not in ``ACT2FN``, or is one of the entries
            ``_DISALLOWED`` rejects. Refusing is the point: the caller has a config
            asking for a function the row cannot compute, and substituting SiLU
            would train a different model than the one described.
    """
    name = activation.value if hasattr(activation, "value") else activation
    if name == "swish":
        name = "silu"
    if name in _DISALLOWED:
        raise ValueError(
            f"Activation {name!r} cannot run through a gated-MLP row: it {_DISALLOWED[name]}. "
            f"Leave the model on its `eager` projection fallback, which calls the model's own "
            f"`act_fn` module."
        )
    if _resolver(name) is None:
        raise ValueError(
            f"Unsupported MLP activation {name!r}: it is not in `transformers.activations.ACT2FN`. "
            f"The rows evaluate {len(supported_activations())} names from that table, excluding "
            f"{sorted(_DISALLOWED)}. Select a row that evaluates this activation, or leave the "
            f"model on its `eager` projection fallback."
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
