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
Patch configuration for KimiK3 NPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.kimi_k3.kimi_k3_npu_patch_gen_config -o veomni/models/transformers/kimi_k3/generated --diff

Re-registers every patch from the GPU config against an NPU-target file so the
NPU modeling is generated from the same source (the eager fallbacks in the
vendored source module — kda_fallback, eager MLA attention — are NPU-safe, and
the VeomniOp handles resolve per ops config).
"""

from veomni.models.transformers.kimi_k3.kimi_k3_gpu_patch_gen_config import (
    PatchedKimiK3Experts,
    PatchedKimiK3SparseMoeBlock,
    kimi_k3_for_conditional_generation_forward_patched,
    kimi_k3_for_conditional_generation_init_patched,
    kimi_k3_get_dyn_bsz_length_fn_patched,
    kimi_k3_get_extra_collate_infos_patched,
    kimi_k3_get_parallel_plan_patched,
    kimi_k3_get_position_id_func_patched,
    kimi_k3_pretrained_init_weights_patched,
    kimi_k3_tie_weights_patched,
    kimi_linear_for_causal_lm_forward_patched,
    kimi_linear_for_causal_lm_init_patched,
    kimi_linear_get_parallel_plan_patched,
    kimi_linear_model_init_patched,
    kimi_moe_gate_forward_patched,
    kimi_pretrained_init_weights_patched,
    kimi_rmsnorm_forward_patched,
    kimi_rmsnorm_init_patched,
    situ_and_mul_forward_patched,
    situ_and_mul_init_patched,
)
from veomni.models.transformers.kimi_k3.kimi_k3_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="veomni.models.transformers.kimi_k3.source.modeling_kimi_k3",
    target_file="patched_modeling_kimi_k3_npu.py",
    description="KimiK3 with NPU-safe eager fallbacks and VeOmni fused-MoE / fused-loss patches",
)

# Mirror additional imports + post-import helpers from the GPU config so the
# generated file is self-contained (same VeomniOp imports, same dropped
# imports).
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.drop_imported_names.update(gpu_config.drop_imported_names)


# ── RMSNorm ──────────────────────────────────────────────────────────────────


config.override_method(
    "KimiRMSNorm.__init__",
    replacement=kimi_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniOp",
)


config.override_method(
    "KimiRMSNorm.forward",
    replacement=kimi_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniOp",
)

# ── SituGLU (fused Triton kernel) ────────────────────────────────────────────


config.override_method(
    "SituAndMul.__init__",
    replacement=situ_and_mul_init_patched,
    description="Construct a local situ_and_mul VeomniOp",
)


config.override_method(
    "SituAndMul.forward",
    replacement=situ_and_mul_forward_patched,
    description="Always call the local situ_and_mul VeomniOp",
)


# ── MoE gate (training-mode routing) ─────────────────────────────────────────


config.override_method(
    "KimiMoEGate.forward",
    replacement=kimi_moe_gate_forward_patched,
    description="Drop the inference-only `assert not self.training` so routing runs in training mode",
)

# ── Routed experts / sparse MoE block ────────────────────────────────────────


config.replace_class(
    "KimiBlockSparseMLP",
    replacement=PatchedKimiK3Experts,
    description=(
        "Use v5 gate_up_proj expert weights ([E, 2I, H] F.linear layout) and the "
        "instance-local moe_experts VeomniOp (kimi_k3 variant for SituGLU, standard otherwise)."
    ),
)


config.replace_class(
    "KimiSparseMoeBlock",
    replacement=PatchedKimiK3SparseMoeBlock,
    description="Training-mode forward: fused experts + latent-MoE projections + shared experts",
)

# ── Inits ────────────────────────────────────────────────────────────────────


config.override_method(
    "KimiLinearModel.__init__",
    replacement=kimi_linear_model_init_patched,
    description="Respect the requested `_attn_implementation` instead of forcing flash_attention_2",
)


config.override_method(
    "KimiK3ForConditionalGeneration.__init__",
    replacement=kimi_k3_for_conditional_generation_init_patched,
    description=(
        "Propagate the top-level `_attn_implementation` to text_config and vision_config "
        "before construction (vision 'sdpa' falls back to 'eager')"
    ),
)


config.override_method(
    "KimiK3ForConditionalGeneration.tie_weights",
    replacement=kimi_k3_tie_weights_patched,
    description="Accept the transformers v5 tie_weights kwargs (missing_keys / recompute_mapping) in the override",
)

# ── Forwards (fused CE) ──────────────────────────────────────────────────────


config.override_method(
    "KimiLinearForCausalLM.__init__",
    replacement=kimi_linear_for_causal_lm_init_patched,
    description="Bind ForCausalLMLoss VeomniOp on KimiLinearForCausalLM",
)


config.override_method(
    "KimiLinearForCausalLM.forward",
    replacement=kimi_linear_for_causal_lm_forward_patched,
    description="Fused cross-entropy path via the VeomniOp-bound loss_function",
)


config.override_method(
    "KimiK3ForConditionalGeneration.forward",
    replacement=kimi_k3_for_conditional_generation_forward_patched,
    description=(
        "Fused cross-entropy path. Calls language_model.model directly so the last hidden "
        "state is always available, then applies language_model.lm_head itself."
    ),
)

# ── Weight init ──────────────────────────────────────────────────────────────


config.override_method(
    "KimiPreTrainedModel._init_weights",
    replacement=kimi_pretrained_init_weights_patched,
    description="Init fused expert weights (in-place, NPU-safe) and zero the KDA dt_bias",
)


config.override_method(
    "KimiK3PreTrainedModel._init_weights",
    replacement=kimi_k3_pretrained_init_weights_patched,
    description="Init fused expert weights (in-place, NPU-safe) and zero the KDA dt_bias",
)

# ── Parallel plans ───────────────────────────────────────────────────────────


config.override_method(
    "KimiK3ForConditionalGeneration.get_position_id_func",
    replacement=kimi_k3_get_position_id_func_patched,
    description="VLM trainer data hook (arange stub; positions are model-internal)",
)

config.override_method(
    "KimiK3ForConditionalGeneration.get_extra_collate_infos",
    replacement=kimi_k3_get_extra_collate_infos_patched,
    description="K3 SP contract: redundant vision per SP rank (no sp-slice on pixel_values/grid_thws)",
)

config.override_method(
    "KimiK3ForConditionalGeneration.get_dyn_bsz_length_fn",
    replacement=kimi_k3_get_dyn_bsz_length_fn_patched,
    description="Dynamic-batching length: add model-side-expanded vision tokens to the pack budget",
)


config.override_method(
    "KimiLinearForCausalLM.get_parallel_plan",
    replacement=kimi_linear_get_parallel_plan_patched,
    description="Register KimiLinear expert parallel plan for v5 generated modeling",
)


config.override_method(
    "KimiK3ForConditionalGeneration.get_parallel_plan",
    replacement=kimi_k3_get_parallel_plan_patched,
    description="Register KimiK3 expert parallel plan for v5 generated modeling",
)
