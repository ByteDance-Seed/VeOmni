# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Patch configuration for DeepseekV3 NPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.deepseek_v3.deepseek_v3_npu_patch_gen_config -o veomni/models/transformers/deepseek_v3/generated --diff

NPU differs from GPU in leaf-op kernels only:
1. RMSNorm.forward uses ``veomni.ops.kernels.rms_norm.npu.rms_norm_forward_npu``.
2. ``apply_rotary_pos_emb`` uses ``veomni.ops.kernels.rotary.npu.apply_rotary_pos_emb_npu``.

Structural patches (OpSlot-guarded fused MoE, router autocast, OpSlot-guarded
fused CE returning ``CausalLMOutputWithLogProbs``, parallel plan) are reused
from the GPU config to guarantee GPU/NPU parity.
"""

import torch

from veomni.patchgen.patch_spec import PatchConfig

from .deepseek_v3_gpu_patch_gen_config import (
    PatchedDeepseekV3NaiveMoe,
    deepseek_v3_forcausallm_forward_patched,
    deepseek_v3_get_parallel_plan_patched,
    deepseek_v3_moe_forward_patched,
    deepseek_v3_topk_router_forward_patched,
)


config = PatchConfig(
    source_module="transformers.models.deepseek_v3.modeling_deepseek_v3",
    target_file="patched_modeling_deepseek_v3_npu.py",
    description="DeepseekV3 with NPU fused RMSNorm / RoPE and VeOmni fused-MoE + OpSlot fused-CE patches",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import("veomni.utils.moe_monitor", names=["record_router_indices"])

# Surface ``CausalLMOutputWithLogProbs`` in the generated file so the patched
# ``forward`` (reused from the GPU config) can return per-token log-probs in
# the unified output dataclass.
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)

# Mirror the GPU config's OpSlot declarations: the patched experts.forward and
# ForCausalLM.forward look these up as module-global names, so the generated
# NPU file must expose them at module scope too.
config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    """
)


# ================================================================
# Patch: DeepseekV3RMSNorm.forward
# 1. Use NPU fused RMSNorm kernel.
# ================================================================
@config.override_method(
    "DeepseekV3RMSNorm.forward",
    description="Use NPU fused RMSNorm kernel",
)
def deepseek_v3_rms_norm_forward_npu(self, hidden_states):
    from veomni.ops.kernels.rms_norm.npu import rms_norm_forward_npu

    return rms_norm_forward_npu(self, hidden_states)


# ================================================================
# Patch: apply_rotary_pos_emb
# 1. Use NPU fused rotary embedding kernel.
# ================================================================
@config.replace_function(
    "apply_rotary_pos_emb",
    description="Use NPU fused rotary embedding kernel",
)
def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    from veomni.ops.kernels.rotary.npu import apply_rotary_pos_emb_npu as _apply_rotary_pos_emb_npu

    return _apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)


# ================================================================
# Structural patches reused from GPU config (fused MoE / router autocast /
# fused CE / parallel plan). Keeping these aligned across backends ensures
# checkpoint interchangeability between GPU and NPU runs.
# ================================================================
config.replace_class(
    "DeepseekV3NaiveMoe",
    replacement=PatchedDeepseekV3NaiveMoe,
    description="Use v5 gate_up_proj expert layout with OpSlot-guarded VeOmni fused-MoE path",
)

config.override_method(
    "DeepseekV3TopkRouter.forward",
    replacement=deepseek_v3_topk_router_forward_patched,
    description="Disable autocast around fp32 router linear for VeRL actor/rollout parity",
)

config.override_method(
    "DeepseekV3MoE.forward",
    replacement=deepseek_v3_moe_forward_patched,
    description="Report top-k indices to the MoE load-balance monitor",
)

config.override_method(
    "DeepseekV3ForCausalLM.forward",
    replacement=deepseek_v3_forcausallm_forward_patched,
    description="OpSlot guard for fused cross entropy in DeepseekV3ForCausalLM.forward",
)

config.override_method(
    "DeepseekV3ForCausalLM.get_parallel_plan",
    replacement=deepseek_v3_get_parallel_plan_patched,
    description="Register DeepseekV3 expert parallel plan for v5 generated modeling",
)


_ = (torch,)
