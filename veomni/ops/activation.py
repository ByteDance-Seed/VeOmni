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

Both ``gated_mlp`` and ``moe_experts`` compute ``act_fn(gate) * up``. Which
``act_fn`` is a property of the *model*, not of the row: a config says
``hidden_act: gelu`` and expects its MLP to use GELU whether or not the row
quantizes. Hard-coding SiLU inside a row made that impossible, so a model with
another activation fell back to its projection modules -- and a quantizing row
that fell back trained at full precision while the config claimed otherwise.

This module is deliberately pure PyTorch, with no kernel of its own:

* :func:`normalize_activation` / :func:`supports` let a row validate a selection
  without knowing the config's vocabulary.
* :func:`apply_activation` evaluates one.

**Every name delegates to the transformer's own implementation**, looked up by the
name the config used. That is not redundancy: the names are not interchangeable.
``gelu_new`` resolves to ``NewGELUActivation``, which stages ``torch.pow(x, 3.0)``,
while ``gelu_pytorch_tanh`` resolves to ``PytorchGELUTanh``, which calls
``F.gelu(approximate="tanh")`` and writes ``x * x * x`` internally. In bf16 the
intermediates round differently -- max ~1.6e-2 on unit-scale inputs -- so mapping
them onto one formula would change a model's outputs and gradients against the
function it trained with.

The derivative is deliberately absent. Every row is an opaque ``wrapper`` now, so
autograd differentiates whatever the row computes and there is no hand-written
backward to keep in sync -- which is also what keeps activation offload and
``torch.compile(fullgraph=True)`` working (``torch.func.vjp`` refuses to run inside
``saved_tensors_hooks``).

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

import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "SUPPORTED_ACTIVATIONS",
    "apply_activation",
    "normalize_activation",
    "supported_activations",
    "supports",
]


#: Activations the gated-MLP rows evaluate, as a config spells them.
#:
#: An explicit list, not a derivation from ``ACT2FN``. An earlier revision did
#: derive the set, and the review that asked for it made a fair point: a
#: hand-maintained list drifts from the library it mirrors. That concern does not
#: apply to a list whose purpose is to *restrict*. Deriving means every name
#: transformers ships becomes a capability this repository claims, and every
#: ``ACT2FN`` entry is a function some model might be configured with -- including
#: ones whose backward cannot be reproduced here (``mish``; see below).
#:
#: The set is bounded by what the repository actually runs. Scanning all 36 toy
#: models, ``hidden_act`` / ``hidden_activation`` take exactly two values: ``silu``
#: and ``gelu_pytorch_tanh``. The latter only appears on Gemma3 and on vision
#: towers, and neither reaches an ``activation=`` call -- Gemma3 binds the ``geglu``
#: variant, whose rows take no such argument and are fixed to GELU-tanh, and the
#: vision towers bind no ``gated_mlp`` at all. So on the ``standard`` variant the
#: only activation any model passes today is ``silu``.
#:
#: What is kept beyond that, and why:
#:
#: * ``swish`` -- HF's own alias (``ACT2FN["swish"] is ACT2FN["silu"]``), used in
#:   the repository's own configs; refusing it would reject a value that is
#:   bit-for-bit the row's default.
#: * ``gelu`` -- the exact-erf form, the most likely hand-edited value.
#: * ``gelu_pytorch_tanh`` / ``gelu_new`` -- two *different* implementations
#:   (``torch.pow(x, 3.0)`` versus ``x * x * x``), which is the distinction that
#:   produced a real bug earlier in this review. Keeping both keeps that pinned.
#:
#: Everything else -- ``gelu_10``, ``gelu_accurate``, ``gelu_fast``,
#: ``gelu_python``, ``gelu_python_tanh``, ``hardswish``, ``laplace``,
#: ``leaky_relu``, ``linear``, ``quick_gelu``, ``relu``, ``relu2``, ``relu6``,
#: ``sigmoid``, ``sqrtsoftplus``, ``tanh`` -- is an unexercised capability, so no
#: row claims it. The reference ``eager`` row still runs such a model exactly: it
#: calls the block's own ``act_fn`` through the projection modules, which is the
#: expression the model used before this family existed. A *quantizing* row
#: refuses instead of falling back (see ``veomni.ops.kernels.gated_mlp._module``).
#: The ``geglu`` variant is a third case: it is fixed to
#: ``gelu_pytorch_tanh`` / ``gelu_python_tanh`` and falls back to ``act_fn`` for
#: anything else, whether or not the name is in this set.
SUPPORTED_ACTIVATIONS: tuple[str, ...] = (
    "silu",
    "swish",
    "gelu",
    "gelu_new",
    "gelu_pytorch_tanh",
)


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
    """Every config-level activation name the gated-MLP rows evaluate.

    See :data:`SUPPORTED_ACTIVATIONS` for why this is an explicit list.
    """
    return frozenset(SUPPORTED_ACTIVATIONS)


def supports(activation: str) -> bool:
    """Whether this module can evaluate *activation* on this host."""
    return activation in supported_activations()


def normalize_activation(activation: str) -> str:
    """Return the canonical spelling of *activation*.

    Raises:
        ValueError: if the name is outside :data:`SUPPORTED_ACTIVATIONS`. Refusing
            is the point: the caller has a config asking for a function the row does
            not claim to compute, and substituting SiLU would train a different
            model than the one described. The caller's projection fallback handles
            it exactly, for any activation.
    """
    name = activation.value if hasattr(activation, "value") else activation
    if name == "swish":
        name = "silu"
    if name not in supported_activations():
        raise ValueError(
            f"Unsupported MLP activation {name!r}. The gated-MLP rows evaluate "
            f"{sorted(supported_activations())}. Any other activation takes the model's own "
            f"`eager` projection fallback, which calls its `act_fn` and is exact; a quantizing "
            f"row will refuse instead."
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
