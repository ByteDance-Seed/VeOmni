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

"""Projection-module resolution for the ``linear`` family.

A row receives *modules*, not weight tensors, so the model writes
``self.veomni_q_a_proj(self.q_a_proj, x)`` and the row decides how to reach the
weight. That keeps the decision in one place: whether a projection can be driven
by its weight, or has to run its own ``forward``, is a property of the row, not
of the call site.

Adapters are the reason the distinction exists. A wrapped projection such as
``LoraLinear`` keeps the base layer at ``base_layer`` and owns no ``weight`` of
its own, so a row that reads ``module.weight`` cannot run on it. A row that does
not quantize simply calls the module and the adapter runs normally; a row that
does quantize refuses, because calling the wrapped module would silently train
at full precision while the config still claimed otherwise.
"""

from __future__ import annotations

from torch import Tensor, nn


__all__ = [
    "QUANTIZING_IMPLS",
    "resolve_projection",
]

#: ``linear`` rows that fake-quantize the weight they are handed, and therefore
#: cannot fall back to calling a wrapped projection module.
QUANTIZING_IMPLS = frozenset({"qat", "torchao", "deepgemm"})


def resolve_projection(
    op_impl: str,
    module: nn.Module,
    x: Tensor,
    *,
    allow_linear_subclass: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    """Return ``(weight, output)`` for a projection, exactly one of which is set.

    Args:
        op_impl: The resolved implementation name of the calling row.
        module: The projection the row was asked to run.
        x: Input activation, used only on the fallback path.
        allow_linear_subclass: Allow ``nn.Linear`` subclasses whose custom
            ``forward`` matches the row contract (used by the grouped QAT row for
            ``DeepseekV4GroupedLinear``).

    Returns:
        ``(module.weight, None)`` when the module owns a weight the row may read,
        otherwise ``(None, module(x))`` -- the module's own forward.

    Raises:
        ValueError: when the module is wrapped or is an ``nn.Linear`` subclass
            whose own ``forward`` computes extra math or unsharding (e.g.
            ``FSDPLinear`` from ``basic_modules``) while the row quantizes.
            Reading ``module.weight`` would bypass the module's ``forward``,
            and calling the fallback would train at full precision under a
            low-precision config.
    """
    if type(module) is nn.Linear:
        return module.weight, None
    if allow_linear_subclass and isinstance(module, nn.Linear):
        return module.weight, None
    if op_impl in QUANTIZING_IMPLS:
        if isinstance(module, nn.Linear):
            raise ValueError(
                f"linear_implementation={op_impl!r} cannot run on nn.Linear subclass {type(module).__name__!r}: "
                "its forward does more than F.linear on its weight (for example, a projection listed in "
                "FSDP2 `basic_modules` becomes FSDPLinear and unshards its weight there), so reading "
                "`module.weight` would bypass it. Set linear_implementation='eager', or drop the overlapping "
                "`basic_modules` entry."
            )
        raise ValueError(
            f"linear_implementation={op_impl!r} cannot run on {type(module).__name__!r}: the row "
            "rounds the projection weight, and a wrapped projection (LoRA keeps the base "
            "layer at `base_layer`) does not expose one. Set linear_implementation='eager' for "
            "adapter training on this projection."
        )
    return None, module(x)
