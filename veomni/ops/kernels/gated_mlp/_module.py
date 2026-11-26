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

"""How a ``gated_mlp`` row reaches a model's projections.

Rows receive the **MLP module**, not six weight tensors, for the same reason the
``linear`` rows do: the module is what knows how its projections may be run.

A row cannot serve every call, and the reasons are all properties of the module or
of the row rather than of the call site:

* **A wrapped projection.** LoRA replaces ``gate_proj`` with a ``LoraLinear`` that
  keeps the real layer at ``base_layer``, so ``gate_proj.weight`` does not exist.
* **An ``nn.Linear`` subclass.** A projection listed in FSDP2 ``basic_modules``
  becomes ``FSDPLinear`` and unshards inside its own ``forward``, or a custom
  subclass computes extra math there, so reading its ``weight`` directly would
  bypass it.
* **An activation the row does not implement.** ``liger_kernel`` calls Liger's
  ``swiglu_forward``, which is SiLU only.

Each of those has one answer: run the projection modules instead, which is what the
model itself used before this family existed. :func:`render` makes that decision,
so a model's MLP forward is a single handle call with no branch.

What "run the projection modules" means differs by variant, and each row owns it:

* ``standard`` evaluates the activation itself through ``veomni.ops.activation``,
  and for a name outside that set calls the block's own ``act_fn`` -- the
  expression the model used before this family, exact for any activation.
* ``geglu`` computes ``gelu_pytorch_tanh`` and calls ``act_fn`` for anything else.

A **quantizing** row has no such answer: falling back to the modules would train at
full precision while the config claimed otherwise, so :func:`render` raises there.
"""

from __future__ import annotations

from torch import nn


__all__ = [
    "QUANTIZING_IMPLS",
    "ROW_ACTIVATIONS",
    "Rendered",
    "plain_projections",
    "render",
]


#: Rows that quantize the weights they are handed. Only these require a real
#: ``weight``; every other row is arithmetic that also works on projection modules.
#: ``torchao`` serves FP8, NVFP4, and MXFP4; the format is its ``dtype`` argument
#: and does not change whether it quantizes.
QUANTIZING_IMPLS = frozenset({"qat", "torchao", "deepgemm"})

#: Activations each row implements, beyond the module shape check. A row absent
#: here is held to ``eager``'s behaviour -- an unknown backend's activation support
#: cannot be assumed. ``None`` means "every activation ``veomni.ops.activation``
#: evaluates".
ROW_ACTIVATIONS: dict[str, frozenset[str] | None] = {
    "eager": None,
    "qat": None,
    "torchao": None,
    # Its Triton kernel fuses ``aten.silu`` into the grouped GEMM in both
    # directions, so it cannot evaluate another activation.
    "deepgemm": frozenset({"silu", "swish"}),
    "liger_kernel": frozenset({"silu"}),
}


def plain_projections(mlp: nn.Module) -> bool:
    """Whether all three projections are exact ``nn.Linear`` layers.

    The exact type matters, not ``isinstance``: an ``nn.Linear`` subclass (such
    as ``FSDPLinear`` from ``basic_modules``) does extra work inside ``forward``,
    so reading its ``weight`` directly would bypass it.
    """
    return all(type(getattr(mlp, name)) is nn.Linear for name in ("gate_proj", "up_proj", "down_proj"))


class Rendered:
    """What a row should run with: six weight tensors, or the projection modules.

    Exactly one of ``weights`` and ``projections`` is set. ``weights`` unreadable
    and the row quantizing is a raise, not a fallback, so the two are mutually
    exclusive by construction.
    """

    __slots__ = ("weights", "projections")

    def __init__(self, weights, projections):
        self.weights = weights
        self.projections = projections


def render(mlp: nn.Module, *, impl: str | None, activation: str) -> Rendered:
    """Decide how *impl* runs this MLP: by weight, or through its projections.

    Args:
        mlp: The MLP block owning ``gate_proj`` / ``up_proj`` / ``down_proj``.
        impl: The row's resolved implementation name.
        activation: The model's ``hidden_act``.

    Raises:
        ValueError: when a fallback is needed and the row fake-quantizes. Falling
            back there would train at full precision under a config that said
            otherwise -- the silent failure this family exists to prevent.

    Returns:
        A :class:`Rendered` holding either the weight tensors (with an empty tensor
        for an unused bias) or the projection modules.
    """
    from ...activation import normalize_activation

    try:
        # Canonical spelling first: ``swish`` is ``ACT2FN``'s alias for SiLU, so a
        # row that implements ``silu`` implements it too. Comparing raw config
        # strings would send those models down the projection path for nothing.
        name = normalize_activation(activation)
    except ValueError:
        # An activation nothing here evaluates; a row cannot serve it either.
        name = None
    plain = plain_projections(mlp)
    allowed = ROW_ACTIVATIONS.get(impl, frozenset({"silu"})) if impl else frozenset({"silu"})
    served = name is not None and (allowed is None or name in allowed)

    if plain and served:
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        empty = gate.weight.new_empty(0)
        return Rendered(
            weights=(
                gate.weight,
                gate.bias if gate.bias is not None else empty,
                up.weight,
                up.bias if up.bias is not None else empty,
                down.weight,
                down.bias if down.bias is not None else empty,
            ),
            projections=None,
        )

    if impl in QUANTIZING_IMPLS:
        if not plain:
            raise ValueError(
                f"gated_mlp_implementation={impl!r} cannot run on this MLP: it quantizes the "
                "projection weights it is handed, and `gate_proj` / `up_proj` / `down_proj` are "
                "either wrapped (LoRA keeps the base weights at `base_layer`) or an nn.Linear "
                "subclass that does extra work in forward. Set "
                "gated_mlp_implementation='eager' for adapter training."
            )
        raise ValueError(
            f"gated_mlp_implementation={impl!r} cannot run on hidden_act={activation!r}: no "
            "quantizing row evaluates that activation, and falling back would train at full "
            "precision. Set gated_mlp_implementation='eager' to keep the projection path."
        )
    return Rendered(weights=None, projections=(mlp.gate_proj, mlp.up_proj, mlp.down_proj))
