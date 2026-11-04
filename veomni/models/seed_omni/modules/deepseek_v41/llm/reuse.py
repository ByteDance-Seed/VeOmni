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

"""The one place V4.1 reaches into ``deepseek_v4``'s patchgen output.

The generated modeling comes in a GPU and an NPU variant and the choice is made
by import, so every reused class has to be resolved through a single switch or
half the module family ends up pinned to the wrong accelerator. Routing them all
through here also makes the reuse surface greppable: what this file re-exports
*is* the list of V4 classes V4.1 shares, and anything not here was rewritten.

``device_patch.apply_veomni_deepseek_v4_device_patch`` is deliberately **not**
called. It swaps the generated module's global ``apply_rotary_pos_emb`` for a
backend chosen from the ops registry, and V4 disables the registry's own default
(``liger_kernel``) because that kernel cannot do partial interleaved rotary --
which is why V4 YAMLs have to pin ``rotary_pos_emb_implementation``. Calling it
from an import would make importing this package fail under any config that has
not pinned that field. V4.1 rotates locally instead; see
``attention.apply_interleaved_rope``.
"""

from typing import Any

from ......models.checkpoint.convert import split_moe_params
from ......models.transformers.deepseek_v4.checkpoint_tensor_converter import (
    validate_deepseek_v4_export_quant_support,
)
from ......models.transformers.deepseek_v4.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    dense_sparse_attn,
    main_window_indices,
)
from ......ops import VeomniOp
from ......ops.config import resolve_op_impl, resolve_qat_impl
from ......utils.device import IS_NPU_AVAILABLE


if IS_NPU_AVAILABLE:
    from ......models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as _generated
else:
    from ......models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as _generated


def resolve_module_op_impl(config: Any, field: str, *, variant: str | None = None) -> str | None:
    """Resolve one op field from ``config.ops_implementation``, falling back to ``resolve_op_impl``."""
    ops_impl = getattr(config, "ops_implementation", None)
    if isinstance(ops_impl, dict) and field in ops_impl and ops_impl[field] is not None:
        impl = ops_impl[field]
        if variant is not None:
            from ......arguments.arguments_types import _DEPRECATED_VARIANT_IMPLEMENTATION_ALIASES

            alias = _DEPRECATED_VARIANT_IMPLEMENTATION_ALIASES.get(field, {}).get(variant, {}).get(impl)
            if alias is not None:
                impl = alias
        return impl
    return resolve_op_impl(field, variant=variant)


def resolve_module_qat_impl(config: Any) -> str:
    """Resolve ``qat_implementation`` from ``config.ops_implementation``, falling back to ``resolve_qat_impl``."""
    ops_impl = getattr(config, "ops_implementation", None)
    if isinstance(ops_impl, dict) and "qat_implementation" in ops_impl and ops_impl["qat_implementation"] is not None:
        return ops_impl["qat_implementation"]
    return resolve_qat_impl()


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
    "resolve_module_qat_impl",
    "split_moe_params",
    "validate_deepseek_v4_export_quant_support",
]
