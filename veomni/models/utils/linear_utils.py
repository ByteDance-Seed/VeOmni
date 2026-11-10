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

"""Helpers for handing a projection's ``weight`` and ``bias`` to a linear op.

Registered linear rows take tensors, not modules, so a model that owns its
projections passes the parameters through. An unused bias is the empty-tensor
sentinel the ``linear`` and ``swiglu_mlp`` families already document, which
keeps every row's signature free of ``Optional``.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn


#: ``linear`` rows that fake-quantize the weight they are handed, and therefore
#: cannot fall back to calling a wrapped projection module.
QUANTIZING_LINEAR_IMPLS = frozenset({"qat"})


__all__ = [
    "QUANTIZING_LINEAR_IMPLS",
    "has_plain_projections",
    "optional_bias",
    "project_through_op",
]


def optional_bias(linear: nn.Module) -> Tensor:
    """Return ``linear``'s bias, or the empty sentinel tensor when it has none."""
    return linear.weight.new_empty(0) if linear.bias is None else linear.bias


def has_plain_projections(module: nn.Module, names: Iterable[str]) -> bool:
    """Whether every named projection on ``module`` is a plain ``nn.Linear``.

    A row that takes a projection's weight as a tensor can only be used on a
    module that still owns one. LoRA replaces a matched ``nn.Linear`` with a
    ``LoraLinear`` holding the base layer at ``base_layer``, which has no
    ``weight``, so such a model has to call the projection modules themselves.
    This is the same rule ``veomni.ops.kernels.swiglu_mlp.has_plain_linear_projections``
    applies to ``gate_proj``/``up_proj``/``down_proj``.
    """
    return all(isinstance(getattr(module, name, None), nn.Linear) for name in names)


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
