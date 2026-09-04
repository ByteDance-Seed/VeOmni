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
Patch configuration for Qwen3-VL-MoE NPU VeomniKernel replacements.

Inherits every GPU patch from `qwen3_vl_moe_gpu_patch_gen_config`.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3_vl_moe.qwen3_vl_moe_npu_patch_gen_config -o veomni/models_kernel/transformers/qwen3_vl_moe/generated --diff
"""

from veomni.models_kernel.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    apply_rotary_pos_emb_patched,
    apply_rotary_pos_emb_vision_patched,
    qwen3_vl_get_metadata_collate_func_patched,
    qwen3_vl_get_position_id_func_patched,
    qwen3_vl_model_get_image_features_patched,
    qwen3_vl_model_get_placeholder_mask_patched,
    qwen3_vl_rmsnorm_forward_patched,
    qwen3_vl_rmsnorm_init_patched,
    qwen3_vl_text_attention_forward_patched,
    qwen3_vl_text_deepstack_process_patched,
    qwen3_vl_vision_attention_forward_patched,
    qwen3_vl_vision_block_forward_patched,
    qwen3_vl_vision_dummy_forward_patched,
    qwen3_vl_vision_fast_pos_embed_interpolate_patched,
    qwen3_vl_vision_forward_patched,
    qwen3_vl_vision_rot_pos_emb_patched,
)
from veomni.models_kernel.transformers.qwen3_vl_moe.qwen3_vl_moe_gpu_patch_gen_config import (
    PatchedQwen3VLMoeTextExperts,
    qwen3_vl_moe_for_conditional_generation_forward_patched,
    qwen3_vl_moe_for_conditional_generation_init_patched,
    qwen3_vl_moe_get_parallel_plan_patched,
    qwen3_vl_moe_model_forward_patched,
)
from veomni.models_kernel.transformers.qwen3_vl_moe.qwen3_vl_moe_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
    target_file="patched_modeling_qwen3_vl_moe_npu.py",
    description="Qwen3-VL-MoE with VeOmni v5 patches and VeomniKernel NPU replacements",
)

# Mirror additional imports + post-import helpers from the GPU config so the
# generated file is self-contained (same SP helpers, same rot_pos_ids /
# async ulysses / get_position_id helpers).
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
# Propagate the GPU config's dropped imports (e.g. ``Qwen3VLMoeCausalLMOutputWithPast``,
# now superseded by ``Qwen3VLMoeCausalLMOutputWithLogProbs`` for the FSDP2-safe
# pre-backward unshard hook on ``lm_head``).
config.drop_imported_names.update(gpu_config.drop_imported_names)


# ================================================================
# Shared VLM + MoE patches (same as GPU config)
# ================================================================
_NAME_MAP = {"Qwen3VL": "Qwen3VLMoe"}
config.override_method(
    "Qwen3VLMoeTextRMSNorm.__init__",
    replacement=qwen3_vl_rmsnorm_init_patched,
    name_map=_NAME_MAP,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3VLMoeTextRMSNorm.forward",
    replacement=qwen3_vl_rmsnorm_forward_patched,
    name_map=_NAME_MAP,
    description="Always call the local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3VLMoeVisionAttention.forward",
    replacement=qwen3_vl_vision_attention_forward_patched,
    name_map=_NAME_MAP,
    description="Use precomputed max_seqlen passed from outer forward to avoid per-layer CPU-GPU sync",
)
config.override_method(
    "Qwen3VLMoeVisionBlock.forward",
    replacement=qwen3_vl_vision_block_forward_patched,
    name_map=_NAME_MAP,
    description="Propagate precomputed max_seqlen to attention to avoid per-layer CPU-GPU sync",
)
config.override_method(
    "Qwen3VLMoeVisionModel.rot_pos_emb",
    replacement=qwen3_vl_vision_rot_pos_emb_patched,
    name_map=_NAME_MAP,
    description="Use lru_cached rot_pos_ids helper (vllm-style) to avoid per-image Python loops",
)
config.override_method(
    "Qwen3VLMoeVisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_vl_vision_fast_pos_embed_interpolate_patched,
    name_map=_NAME_MAP,
    description="Tensorized meshgrid implementation of fast_pos_embed_interpolate",
)
config.override_method(
    "Qwen3VLMoeVisionModel.forward",
    replacement=qwen3_vl_vision_forward_patched,
    name_map=_NAME_MAP,
    description="VeOmni SP + deepstack + precomputed max_seqlen; return BaseModelOutputWithDeepstackFeatures",
)
config.override_method(
    "Qwen3VLMoeVisionModel.dummy_forward",
    replacement=qwen3_vl_vision_dummy_forward_patched,
    name_map=_NAME_MAP,
    description="Provide dummy vision forward for FSDP path with SP-aware shape",
)
config.override_method(
    "Qwen3VLMoeTextAttention.forward",
    replacement=qwen3_vl_text_attention_forward_patched,
    name_map=_NAME_MAP,
    description="Route through async Ulysses fused QKV/Output projection when async_enabled",
)
config.override_method(
    "Qwen3VLMoeTextModel._deepstack_process",
    replacement=qwen3_vl_text_deepstack_process_patched,
    name_map=_NAME_MAP,
    description="Handle visual_pos_masks=None by adding 0.0 so FSDP sees the visual params",
)
config.override_method(
    "Qwen3VLMoeModel.get_image_features",
    replacement=qwen3_vl_model_get_image_features_patched,
    name_map=_NAME_MAP,
    description="Return flat image_embeds tensor (skip per-image torch.split)",
)
config.override_method(
    "Qwen3VLMoeModel.get_placeholder_mask",
    replacement=qwen3_vl_model_get_placeholder_mask_patched,
    name_map=_NAME_MAP,
    description="Return raw image/video placeholder bool masks for VeOmni SP-aware masked_scatter",
)
config.override_method(
    "Qwen3VLMoeModel.forward",
    replacement=qwen3_vl_moe_model_forward_patched,
    description="VeOmni SP + precomputed position-id + dummy-forward + deepstack; preserve MoE router_logits",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.get_position_id_func",
    replacement=qwen3_vl_get_position_id_func_patched,
    name_map=_NAME_MAP,
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.get_metadata_collate_func",
    replacement=qwen3_vl_get_metadata_collate_func_patched,
    name_map=_NAME_MAP,
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)

# MoE-specific patches reused verbatim from the GPU config.
config.replace_class(
    "Qwen3VLMoeTextExperts",
    replacement=PatchedQwen3VLMoeTextExperts,
    description="Drop @use_experts_implementation and always call moe_experts VeomniKernel",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.__init__",
    replacement=qwen3_vl_moe_for_conditional_generation_init_patched,
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.forward",
    replacement=qwen3_vl_moe_for_conditional_generation_forward_patched,
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.get_parallel_plan",
    replacement=qwen3_vl_moe_get_parallel_plan_patched,
    description="Register Qwen3VLMoe expert parallel plan for v5 generated modeling",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope full VeomniKernel",
)
config.replace_function(
    "apply_rotary_pos_emb_vision",
    replacement=apply_rotary_pos_emb_vision_patched,
    description="Always call rope_vision full VeomniKernel",
)
