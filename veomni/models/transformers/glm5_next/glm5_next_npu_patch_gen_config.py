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
"""
Patch configuration for GLM-5.3-Flash (``glm5_next``) NPU VeomniOp replacements.

Regen command:
patchgen veomni.models.transformers.glm5_next.glm5_next_npu_patch_gen_config -o veomni/models/transformers/glm5_next/generated --diff

Inherits the GPU patch surface. Host defaults select ``torch_npu`` /
``fused_npu`` / ``chunk_loss`` on Ascend; recipes that prefer non-fused ops
should set ``moe_implementation: eager`` explicitly.
"""

from veomni.models.transformers.glm5_next.glm5_next_gpu_patch_gen_config import (
    PatchedGlm5NextTextExperts,
    collate_multimodal_metadata,
    get_glm5_next_position_ids,
    glm5_next_forcg_forward_patched,
    glm5_next_forcg_get_parallel_plan_patched,
    glm5_next_forcg_init_patched,
    glm5_next_get_metadata_collate_func_patched,
    glm5_next_get_position_id_func_patched,
    glm5_next_model_forward_patched,
    glm5_next_model_get_image_features_patched,
    glm5_next_model_get_placeholder_mask_patched,
    glm5_next_model_get_video_features_patched,
    glm5_next_text_decoder_layer_forward_patched,
    glm5_next_text_decoder_layer_init_patched,
    glm5_next_text_get_parallel_plan_patched,
    glm5_next_text_mlp_forward_patched,
    glm5_next_text_mlp_init_patched,
    glm5_next_text_model_init_patched,
    glm5_next_text_moe_forward_patched,
    glm5_next_text_rmsnorm_forward_patched,
    glm5_next_text_rmsnorm_init_patched,
    glm5_next_vision_attention_forward_patched,
    glm5_next_vision_attention_init_patched,
    glm5_next_vision_block_forward_patched,
    glm5_next_vision_block_init_patched,
    glm5_next_vision_dummy_forward_patched,
    glm5_next_vision_forward_patched,
    glm5_next_vision_mlp_forward_patched,
    glm5_next_vision_mlp_init_patched,
    glm5_next_vision_model_init_patched,
    glm5_next_vision_rmsnorm_forward_patched,
    glm5_next_vision_rmsnorm_init_patched,
)
from veomni.models.transformers.glm5_next.glm5_next_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.glm5_next.modeling_glm5_next",
    target_file="patched_modeling_glm5_next_npu.py",
    description="GLM-5.3-Flash NPU with VeomniOp RMS / SwiGLU / MoE / fused loss, EP + Ulysses SP + VLM metadata",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)
# Keep ``rotate_half`` — ``apply_rotary_pos_emb_vision`` depends on it.

config.override_method(
    "Glm5NextTextRMSNorm.__init__",
    replacement=glm5_next_text_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniOp",
)
config.override_method(
    "Glm5NextTextRMSNorm.forward",
    replacement=glm5_next_text_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniOp",
)
config.override_method(
    "Glm5NextRMSNorm.__init__",
    replacement=glm5_next_vision_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniOp for vision",
)
config.override_method(
    "Glm5NextRMSNorm.forward",
    replacement=glm5_next_vision_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniOp for vision",
)
config.override_method(
    "Glm5NextTextMLP.__init__",
    replacement=glm5_next_text_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniOp with swiglu_limit",
)
config.override_method(
    "Glm5NextTextMLP.forward",
    replacement=glm5_next_text_mlp_forward_patched,
    description="Call swiglu_mlp with swiglu_limit for silu/swish",
)
config.override_method(
    "Glm5NextVisionMLP.__init__",
    replacement=glm5_next_vision_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniOp for vision with swiglu_limit",
)
config.override_method(
    "Glm5NextVisionMLP.forward",
    replacement=glm5_next_vision_mlp_forward_patched,
    description="Call swiglu_mlp with swiglu_limit for vision",
)
config.override_method(
    "Glm5NextTextDecoderLayer.__init__",
    replacement=glm5_next_text_decoder_layer_init_patched,
    description="Pass config to Glm5NextTextRMSNorm; keep HF attention/mHC/MoE wiring",
)
config.override_method(
    "Glm5NextTextModel.__init__",
    replacement=glm5_next_text_model_init_patched,
    description="Pass config to final Glm5NextTextRMSNorm",
)
config.override_method(
    "Glm5NextTextDecoderLayer.forward",
    replacement=glm5_next_text_decoder_layer_forward_patched,
    description="Ulysses: all-gather collapsed tokens before attn, slice after; keep HF KDA/DSA/mHC",
)
config.replace_class(
    "Glm5NextTextExperts",
    replacement=PatchedGlm5NextTextExperts,
    description="Always call moe_experts VeomniOp on v5 gate_up_proj with swiglu_limit",
)
config.override_method(
    "Glm5NextTextMoE.forward",
    replacement=glm5_next_text_moe_forward_patched,
    description="Report top-k indices to the MoE load-balance monitor",
)
config.override_method(
    "Glm5NextVisionAttention.__init__",
    replacement=glm5_next_vision_attention_init_patched,
    description="Pass config to vision RMSNorm; bind SP-aware vision attention VeomniOp",
)
config.override_method(
    "Glm5NextVisionAttention.forward",
    replacement=glm5_next_vision_attention_forward_patched,
    description="Call local rope/attention VeomniOps; consume precomputed max_seqlen",
)
config.override_method(
    "Glm5NextVisionBlock.__init__",
    replacement=glm5_next_vision_block_init_patched,
    description="Pass config to vision block RMSNorm constructors",
)
config.override_method(
    "Glm5NextVisionBlock.forward",
    replacement=glm5_next_vision_block_forward_patched,
    description="Propagate precomputed max_seqlen to vision attention",
)
config.override_method(
    "Glm5NextVisionModel.__init__",
    replacement=glm5_next_vision_model_init_patched,
    description="Pass config to vision post_layernorm RMSNorm",
)
config.override_method(
    "Glm5NextVisionModel.forward",
    replacement=glm5_next_vision_forward_patched,
    description="VeOmni SP + precomputed ViT metadata for GLM-5.3-Flash vision",
)
config.override_method(
    "Glm5NextVisionModel.dummy_forward",
    replacement=glm5_next_vision_dummy_forward_patched,
    description="Provide dummy vision forward for FSDP with SP-aware shapes",
)
config.override_method(
    "Glm5NextModel.get_image_features",
    replacement=glm5_next_model_get_image_features_patched,
    description="Return flat image embeds for VeOmni SP-aware masked_scatter",
)
config.override_method(
    "Glm5NextModel.get_video_features",
    replacement=glm5_next_model_get_video_features_patched,
    description="Return flat video embeds; keep HF temporal flatten of grid_thw",
)
config.override_method(
    "Glm5NextModel.get_placeholder_mask",
    replacement=glm5_next_model_get_placeholder_mask_patched,
    description="Return raw image/video placeholder bool masks for VeOmni SP scatter",
)
config.override_method(
    "Glm5NextModel.forward",
    replacement=glm5_next_model_forward_patched,
    description="VeOmni SP + dummy-forward + multimodal_metadata for GLM-5.3-Flash",
)
config.override_method(
    "Glm5NextForConditionalGeneration.get_parallel_plan",
    replacement=glm5_next_forcg_get_parallel_plan_patched,
    description="Register GLM-5.3-Flash expert parallel plan for the VLM wrapper",
)
config.override_method(
    "Glm5NextTextModel.get_parallel_plan",
    replacement=glm5_next_text_get_parallel_plan_patched,
    description="Register GLM-5.3-Flash expert parallel plan for standalone text",
)
config.override_method(
    "Glm5NextForConditionalGeneration.get_position_id_func",
    replacement=glm5_next_get_position_id_func_patched,
    description="Expose standard 1D position ids to the VeOmni VLM data pipeline",
)
config.override_method(
    "Glm5NextForConditionalGeneration.get_metadata_collate_func",
    replacement=glm5_next_get_metadata_collate_func_patched,
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)
config.override_method(
    "Glm5NextForConditionalGeneration.__init__",
    replacement=glm5_next_forcg_init_patched,
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniOps",
)
config.override_method(
    "Glm5NextForConditionalGeneration.forward",
    replacement=glm5_next_forcg_forward_patched,
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniOps; forward VLM kwargs",
)

# Helpers are already extended from gpu_config; keep explicit names referenced.
_ = (collate_multimodal_metadata, get_glm5_next_position_ids)
