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
    "DEEPGEMM_BLOCK_SIZE",
    "QUANTIZING_IMPLS",
    "require_deepgemm_block_size",
    "resolve_projection",
]


#: ``linear`` rows that quantize the weight they are handed, and therefore
#: cannot fall back to calling a wrapped projection module.
#:
#: ``qat`` fake-quantizes both GEMM operands and multiplies in the model dtype;
#: ``deepgemm_fp8`` quantizes them again inside its own FP8 GEMM. Either way the
#: row reads ``module.weight``, so a wrapper with the real weight at
#: ``base_layer`` would leave the config claiming a low-precision GEMM that never
#: ran.
QUANTIZING_IMPLS = frozenset({"qat", "deepgemm_fp8"})

#: The one block width DeepGEMM's blockwise FP8 recipe supports: 128 columns on the
#: reduction axis and 128 rows on the output axis. It is not a tuning knob, so a
#: row that cannot honor another value must refuse it.
DEEPGEMM_BLOCK_SIZE = 128


def require_deepgemm_block_size(block_size: int) -> None:
    """Refuse a ``block_size`` the DeepGEMM rows cannot honor.

    ``linear_implementation`` / ``swiglu_mlp_implementation`` / ``moe_implementation``
    all accept a ``block_size`` keyword, and the ``qat`` rows do use it. On the
    ``deepgemm_fp8`` rows it is fixed, so accepting and ignoring it would let one
    keyword mean two different recipes depending on which row the config selected.
    """
    if block_size != DEEPGEMM_BLOCK_SIZE:
        raise ValueError(
            f"deepgemm_fp8 supports only block_size={DEEPGEMM_BLOCK_SIZE}, got {block_size}. "
            "The block width is part of DeepGEMM's blockwise FP8 recipe, not a per-call choice."
        )


def _is_sharded(weight: Tensor) -> bool:
    """Whether ``weight`` is a ``DTensor``, i.e. a shard rather than the whole tensor.

    ``isinstance(weight, Tensor)`` cannot answer this -- ``DTensor`` *is* a
    ``Tensor`` subclass -- so the type is checked by name to avoid importing it
    (``torch.distributed.tensor`` is not always available).

    Verified against real ``fully_shard`` on gloo, in the two placements a
    projection can have:

    * Sharded as part of its parent block (the normal case): by the time a
      quantizing row is called, the parent's forward has run its unshard hook, so
      ``module.weight`` is a whole ``Parameter`` and this returns ``False``.
    * Listed in ``basic_modules``, i.e. its own FSDP unit: nothing unshards it
      before the row reads it, so ``module.weight`` is the ``DTensor`` and this
      returns ``True`` -- which is the case the row must refuse.

    ``tests/ops/linear/test_linear.py::test_a_quantizing_row_only_refuses_a_projection_that_is_its_own_fsdp_unit``
    pins both.
    """
    return type(weight).__name__ == "DTensor"


def resolve_projection(op_impl: str, module: nn.Module, x: Tensor) -> tuple[Tensor | None, Tensor | None]:
    """Return ``(weight, output)`` for a projection, exactly one of which is set.

    Args:
        op_impl: The resolved implementation name of the calling row.
        module: The projection the row was asked to run.
        x: Input activation, used only on the fallback path.

    Returns:
        ``(module.weight, None)`` when the module owns a weight the row may read,
        otherwise ``(None, module(x))`` -- the module's own forward.

    Raises:
        ValueError: when the module is wrapped while the row fake-quantizes. The
            wrapped module keeps its base layer at ``base_layer``, so the row
            cannot reach the weight it is supposed to round, and running the
            wrapper would train at full precision under a QAT config.
        RuntimeError: when the row quantizes and the projection is currently
            sharded. A quantizing row reads the weight without calling the
            module, so it never runs the unshard hook ``fully_shard`` installs in
            ``forward`` -- see :func:`_is_sharded`.
    """
    weight = getattr(module, "weight", None)
    if isinstance(weight, Tensor):
        if op_impl in QUANTIZING_IMPLS and _is_sharded(weight):
            raise RuntimeError(
                f"linear_implementation={op_impl!r} read a sharded weight from {type(module).__name__!r}: "
                "the row quantizes the projection without calling its `forward`, so the unshard hook "
                "`fully_shard` installs there never runs, and the GEMM would multiply one rank's slice. "
                "Either drop this projection from `basic_modules` (so it is unsharded by its parent "
                "block's forward) or set linear_implementation='eager'."
            )
        return weight, None
    if op_impl in QUANTIZING_IMPLS:
        raise ValueError(
            f"linear_implementation={op_impl!r} cannot run on {type(module).__name__!r}: the row "
            "fake-quantizes the projection weight, and a wrapped projection (LoRA keeps the base "
            "layer at `base_layer`) does not expose one. Set linear_implementation='eager' for "
            "adapter training on this projection."
        )
    return None, module(x)
