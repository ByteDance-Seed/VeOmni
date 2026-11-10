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

"""Helpers for running a model-owned projection through a ``linear`` op row.

Registered linear rows take tensors, not modules, so a model that owns its
projections passes the parameters through. Adapters complicate that: LoRA
replaces the projection, and the row can no longer reach its weight. This module
is the single place that decides what happens then.
"""

from __future__ import annotations

from torch import Tensor, nn


#: ``linear`` rows that fake-quantize the weight they are handed, and therefore
#: cannot fall back to calling a wrapped projection module.
QUANTIZING_LINEAR_IMPLS = frozenset({"qat"})


__all__ = [
    "QUANTIZING_LINEAR_IMPLS",
    "project_through_op",
]


def project_through_op(op, module: nn.Module, x: Tensor, **kwargs) -> Tensor:
    """Run a projection through its ``linear`` op handle.

    A ``linear`` row takes the projection's ``weight`` as a tensor, which a
    wrapped projection such as ``LoraLinear`` does not own -- it keeps the base
    layer at ``base_layer``. Such a projection is called as a module instead, so
    adapter training keeps working.

    Args:
        op: The ``VeomniOp("linear", ...)`` handle bound on the owning module.
        module: The projection the handle was bound for.
        x: Input activation.
        **kwargs: Extra keyword arguments the row takes, such as ``n_groups``.

    Raises:
        ValueError: when ``module`` is wrapped while ``op`` is a quantizing row.
            Forwarding the unquantized base layer would leave the config claiming
            quantization-aware training while the step runs at full precision.
    """
    if hasattr(module, "weight"):
        return op(x, module.weight, module.weight.new_empty(0), **kwargs)
    if op.impl in QUANTIZING_LINEAR_IMPLS:
        raise ValueError(
            f"linear_implementation={op.impl!r} cannot run on a wrapped projection: the row fake-quantizes "
            "the projection weight, and a layer substituted after construction (LoRA) keeps it at "
            "`base_layer`. Set linear_implementation='eager' for adapter training."
        )
    return module(x)
