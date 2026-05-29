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
Patch configuration for Qwen3-VL NPU build (transformers>=5.2.0).

Inherits every GPU patch from `qwen3_vl_gpu_patch_gen_config` and layers NPU
kernel replacements on top (npu_rms_norm, npu_rotary_mul for text + vision).

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_vl.qwen3_vl_npu_patch_gen_config -o veomni/models/transformers/qwen3_vl/generated --diff
"""

from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    apply_rotary_pos_emb_patched,
    apply_rotary_pos_emb_vision_patched,
    qwen3_vl_for_conditional_generation_forward_patched,
    qwen3_vl_get_metadata_collate_func_patched,
    qwen3_vl_get_position_id_func_patched,
    qwen3_vl_model_forward_patched,
    qwen3_vl_model_get_image_features_patched,
    qwen3_vl_model_get_placeholder_mask_patched,
    qwen3_vl_rmsnorm_forward_patched,
    qwen3_vl_text_attention_forward_patched,
    qwen3_vl_text_deepstack_process_patched,
    qwen3_vl_vision_attention_forward_patched,
    qwen3_vl_vision_block_forward_patched,
    qwen3_vl_vision_dummy_forward_patched,
    qwen3_vl_vision_fast_pos_embed_interpolate_patched,
    qwen3_vl_vision_forward_patched,
    qwen3_vl_vision_rot_pos_emb_patched,
)
from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_vl.modeling_qwen3_vl",
    target_file="patched_modeling_qwen3_vl_npu.py",
    description="Qwen3-VL with VeOmni v5 compatibility + NPU fused RMSNorm/RoPE kernels",
)

# Mirror additional imports + post-import helpers from the GPU config so the
# generated file is self-contained (same SP helpers, same rot_pos_ids /
# async ulysses / get_position_id helpers).
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
# Propagate the GPU config's dropped imports (e.g. ``Qwen3VLCausalLMOutputWithPast``,
# now superseded by ``Qwen3VLCausalLMOutputWithLogProbs`` for the FSDP2-safe
# pre-backward unshard hook on ``lm_head``).
config.drop_imported_names.update(gpu_config.drop_imported_names)

# ================================================================
# Shared GPU patches (SP / deepstack / fused-CE / async Ulysses / ...)
# ================================================================
config.override_method(
    "Qwen3VLTextRMSNorm.forward",
    replacement=qwen3_vl_rmsnorm_forward_patched,
    description="OpSlot guard for NPU fused RMSNorm (standard formulation)",
)
config.override_method(
    "Qwen3VLVisionAttention.forward",
    replacement=qwen3_vl_vision_attention_forward_patched,
    description="Use precomputed max_seqlen passed from outer forward to avoid per-layer CPU-GPU sync",
)
config.override_method(
    "Qwen3VLVisionBlock.forward",
    replacement=qwen3_vl_vision_block_forward_patched,
    description="Propagate precomputed max_seqlen to attention to avoid per-layer CPU-GPU sync",
)
config.override_method(
    "Qwen3VLVisionModel.rot_pos_emb",
    replacement=qwen3_vl_vision_rot_pos_emb_patched,
    description="Use lru_cached rot_pos_ids helper (vllm-style) to avoid per-image Python loops",
)
config.override_method(
    "Qwen3VLVisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_vl_vision_fast_pos_embed_interpolate_patched,
    description="Tensorized meshgrid implementation of fast_pos_embed_interpolate",
)
config.override_method(
    "Qwen3VLVisionModel.forward",
    replacement=qwen3_vl_vision_forward_patched,
    description="VeOmni SP + deepstack + precomputed max_seqlen; return BaseModelOutputWithDeepstackFeatures",
)
config.override_method(
    "Qwen3VLVisionModel.dummy_forward",
    replacement=qwen3_vl_vision_dummy_forward_patched,
    description="Provide dummy vision forward for FSDP path with SP-aware shape",
)
config.override_method(
    "Qwen3VLTextAttention.forward",
    replacement=qwen3_vl_text_attention_forward_patched,
    description="Route through async Ulysses fused QKV/Output projection when async_enabled",
)
config.override_method(
    "Qwen3VLTextModel._deepstack_process",
    replacement=qwen3_vl_text_deepstack_process_patched,
    description="Handle visual_pos_masks=None by adding 0.0 so FSDP sees the visual params",
)
config.override_method(
    "Qwen3VLModel.get_image_features",
    replacement=qwen3_vl_model_get_image_features_patched,
    description="Return flat image_embeds tensor (skip per-image torch.split)",
)
config.override_method(
    "Qwen3VLModel.get_placeholder_mask",
    replacement=qwen3_vl_model_get_placeholder_mask_patched,
    description="Return raw image/video placeholder bool masks for VeOmni SP-aware masked_scatter",
)
config.override_method(
    "Qwen3VLModel.forward",
    replacement=qwen3_vl_model_forward_patched,
    description="VeOmni SP + precomputed position-id + dummy-forward + deepstack multimodal patches",
)
config.override_method(
    "Qwen3VLForConditionalGeneration.get_position_id_func",
    replacement=qwen3_vl_get_position_id_func_patched,
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)
config.override_method(
    "Qwen3VLForConditionalGeneration.get_metadata_collate_func",
    replacement=qwen3_vl_get_metadata_collate_func_patched,
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)
config.override_method(
    "Qwen3VLForConditionalGeneration.forward",
    replacement=qwen3_vl_for_conditional_generation_forward_patched,
    description="Use VeOmni unified fused loss_function path",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="OpSlot guard for NPU fused RoPE",
)
config.replace_function(
    "apply_rotary_pos_emb_vision",
    replacement=apply_rotary_pos_emb_vision_patched,
    description="OpSlot guard for NPU fused vision RoPE",
)
