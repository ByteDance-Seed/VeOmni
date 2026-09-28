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

from ......utils.device import IS_NPU_AVAILABLE


if IS_NPU_AVAILABLE:
    from ......models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as _generated
else:
    from ......models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as _generated


#: RMS norm with VeOmni's fused-kernel dispatch. V4.1's ``norm_eps`` is 1e-20
#: against V4's 1e-6, which is a constructor argument rather than a behaviour
#: difference.
DeepseekV4RMSNorm = _generated.DeepseekV4RMSNorm

#: The mHC input norm -- RMS without a learned weight.
DeepseekV4UnweightedRMSNorm = _generated.DeepseekV4UnweightedRMSNorm

#: Block-diagonal grouped projection, ``o_groups`` independent blocks. This is
#: the released ``einsum("bsgd,grd->bsgr", o, wo_a.view(groups, rank, -1))``
#: written as a module, with the same weight layout.
DeepseekV4GroupedLinear = _generated.DeepseekV4GroupedLinear

#: Stacked routed experts: ``gate_up_proj [E, 2I, H]``, ``down_proj [E, H, I]``,
#: the fused-MoE OpSlot path, fp4 weight QAT keyed on ``expert_dtype``, and the
#: ``swiglu_limit`` clamp. V4.1 changes none of it. Reads the config as
#: ``num_local_experts`` / ``intermediate_size`` / ``hidden_act``, which
#: :class:`DeepseekV41LlmConfig` exposes as properties over its own field names.
DeepseekV4Experts = _generated.DeepseekV4Experts

#: The shared expert -- one SwiGLU FFN with the same two-sided clamp on ``up``
#: and one-sided on ``gate``, and FP8 QAT on all three projections.
DeepseekV4MLP = _generated.DeepseekV4MLP

#: Routed-expert / shared-MLP QAT. The helpers close over this slot in the
#: generated module, so ``_bind_veomni_ops`` will not see it unless a V4.1
#: modeling module re-exports the same object (``llm/modeling.py``,
#: ``llm/accelerated/accelerated.py``).
veomni_qat_implementation = _generated.veomni_qat_implementation

#: The fused-MoE guard in :class:`DeepseekV4Experts`, re-exported for the same
#: reason. Unbound, the experts run V4's eager loop, which indexes the stacked
#: slabs with global expert ids and so cannot run on an EP shard; it also skips
#: the fp4 expert QAT, which is wired on the fused path only.
veomni_moe_experts_forward = _generated.veomni_moe_experts_forward


__all__ = [
    "DeepseekV4Experts",
    "DeepseekV4GroupedLinear",
    "DeepseekV4MLP",
    "DeepseekV4RMSNorm",
    "DeepseekV4UnweightedRMSNorm",
    "veomni_moe_experts_forward",
    "veomni_qat_implementation",
]
