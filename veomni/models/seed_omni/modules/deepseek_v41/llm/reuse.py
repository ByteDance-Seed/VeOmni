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

"""The one place V4.1 reaches into ``deepseek_v4``'s standalone modeling.

Every class V4.1 shares is resolved through this module, which makes the reuse
surface greppable: what this file re-exports *is* the list of V4 classes V4.1
shares, and anything not here was rewritten.

Routing them all through here also keeps the import in one place. Note that
``deepseek_v41`` binds V4's classes directly -- it reads the ops config from its
own module config rather than from ``config.ops_config`` -- so the reused
classes are always constructed with a V4.1 config that exposes the ``ops_config``
property.

``apply_rotary_pos_emb`` is **not** swapped through a device patch. V4 disables
the registry's own default (``liger_kernel``) because that kernel cannot do
partial interleaved rotary -- which is why V4 YAMLs have to pin
``rotary_pos_emb_implementation``. V4.1 rotates locally instead; see
``attention.apply_interleaved_rope``.
"""

from typing import Any

from ......models.checkpoint.convert import split_moe_params
from ......models.transformers.deepseek_v4 import modeling_deepseek_v4 as _generated
from ......models.transformers.deepseek_v4.checkpoint_tensor_converter import (
    validate_deepseek_v4_export_quant_support,
)
from ......models.transformers.deepseek_v4.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    dense_sparse_attn,
    main_window_indices,
)
from ......ops import OpsSelection, VeomniOp


def resolve_module_op_impl(config: Any, field: str, *, variant: str | None = None) -> str | OpsSelection | None:
    """Read one op field off this module's own ``config.ops_implementation``.

    ``None`` means the module did not select it, and a ``VeomniOp`` built from
    ``None`` falls through to the host default for the ``(op, variant)`` -- which
    is what the module wants: it names only what the checkpoint's YAML said.

    An explicit ``"none"`` is returned as written, not rewritten to ``None``. It
    is a real row on the families that have one (``kv_round``), and for those the
    row is what says "leave the tensor alone"; collapsing it to ``None`` would
    fall through to the host default, which on ``kv_round`` is ``triton`` and
    *does* round.

    Mapping-form per-site selections (``{name, kwargs, overrides}``) are converted
    to :class:`~veomni.ops.OpsSelection` so ``VeomniOp(..., site=...)`` resolves
    each site against the override rules directly.
    """
    del variant
    ops_impl = getattr(config, "ops_implementation", None)
    if isinstance(ops_impl, dict):
        impl = ops_impl.get(field)
        if isinstance(impl, dict):
            return OpsSelection.from_config_value(impl)
        if impl is not None:
            return impl
    return None


class DeepseekV4RMSNorm(_generated.DeepseekV4RMSNorm):
    """RMS norm with VeOmni's fused-kernel dispatch, respecting module-level ``config.ops_implementation``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, config: Any = None) -> None:
        super().__init__(hidden_size, eps=eps)
        if config is not None:
            self.veomni_rms_norm = VeomniOp(
                "rms_norm",
                "deepseek_v4",
                resolve_module_op_impl(config, "rms_norm_implementation"),
            )


class DeepseekV4UnweightedRMSNorm(_generated.DeepseekV4UnweightedRMSNorm):
    """The mHC input norm -- RMS without a learned weight, respecting module-level ``config.ops_implementation``."""

    def __init__(self, eps: float = 1e-6, config: Any = None) -> None:
        super().__init__(eps=eps)
        if config is not None:
            self.veomni_unweighted_rms_norm = VeomniOp(
                "rms_norm",
                "unweighted",
                resolve_module_op_impl(config, "rms_norm_implementation", variant="unweighted"),
            )


#: Block-diagonal grouped projection, ``o_groups`` independent blocks. This is
#: the released ``einsum("bsgd,grd->bsgr", o, wo_a.view(groups, rank, -1))``
#: written as a module, with the same weight layout.
DeepseekV4GroupedLinear = _generated.DeepseekV4GroupedLinear

#: Stacked routed experts: ``gate_up_proj [E, 2I, H]``, ``down_proj [E, H, I]``,
#: the fused-MoE VeomniOp path, fp4 weight QAT keyed on ``expert_dtype``, and the
#: ``swiglu_limit`` clamp. V4.1 changes none of it. Reads the config as
#: ``num_local_experts`` / ``intermediate_size`` / ``hidden_act``, which
#: :class:`DeepseekV41LlmConfig` exposes as properties over its own field names.
DeepseekV4Experts = _generated.DeepseekV4Experts

#: The shared expert -- one SwiGLU FFN with the same two-sided clamp on ``up``
#: and one-sided on ``gate``, and FP8 QAT on all three projections.
DeepseekV4MLP = _generated.DeepseekV4MLP


__all__ = [
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DeepseekV4Experts",
    "DeepseekV4GroupedLinear",
    "DeepseekV4MLP",
    "DeepseekV4RMSNorm",
    "DeepseekV4UnweightedRMSNorm",
    "dense_sparse_attn",
    "main_window_indices",
    "resolve_module_op_impl",
    "split_moe_params",
    "validate_deepseek_v4_export_quant_support",
]
