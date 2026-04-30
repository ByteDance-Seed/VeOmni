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
Patch configuration for Qwen3-VL-MoE NPU build (transformers>=5.2.0).

Inherits every GPU patch from `qwen3_vl_moe_gpu_patch_gen_config` (which in
turn reuses the qwen3_vl VLM patches) and layers NPU kernel replacements on
top (npu_rms_norm, npu_rotary_mul for text + vision).

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_vl_moe.qwen3_vl_moe_npu_patch_gen_config -o veomni/models/transformers/qwen3_vl_moe/generated --diff
"""

from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    qwen3_vl_get_position_id_func_patched,
    qwen3_vl_model_get_image_features_patched,
    qwen3_vl_model_get_placeholder_mask_patched,
    qwen3_vl_text_attention_forward_patched,
    qwen3_vl_text_deepstack_process_patched,
    qwen3_vl_vision_attention_forward_patched,
    qwen3_vl_vision_block_forward_patched,
    qwen3_vl_vision_dummy_forward_patched,
    qwen3_vl_vision_fast_pos_embed_interpolate_patched,
    qwen3_vl_vision_forward_patched,
    qwen3_vl_vision_rot_pos_emb_patched,
)
from veomni.models.transformers.qwen3_vl_moe.qwen3_vl_moe_gpu_patch_gen_config import (
    PatchedQwen3VLMoeTextExperts,
    qwen3_vl_moe_for_conditional_generation_forward_patched,
    qwen3_vl_moe_get_parallel_plan_patched,
    qwen3_vl_moe_model_forward_patched,
    qwen3_vl_text_rmsnorm_forward_patched,
)
from veomni.models.transformers.qwen3_vl_moe.qwen3_vl_moe_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
    target_file="patched_modeling_qwen3_vl_moe_npu.py",
    description="Qwen3-VL-MoE with VeOmni v5 compatibility + NPU fused RMSNorm/RoPE kernels",
)

# Mirror additional imports + post-import helpers from the GPU config so the
# generated file is self-contained (same SP helpers, same rot_pos_ids /
# async ulysses / get_position_id helpers.
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.add_post_import_block(
    """
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    """
)
config.helpers.extend(gpu_config.helpers)
config.add_import("torch_npu", is_from_import=False)


# ================================================================
# Shared VLM + MoE patches (same as GPU config)
# ================================================================
_NAME_MAP = {"Qwen3VL": "Qwen3VLMoe"}

config.override_method(
    "Qwen3VLMoeTextRMSNorm.forward",
    replacement=qwen3_vl_text_rmsnorm_forward_patched,
    name_map=_NAME_MAP,
    description="OpSlot guard for fused RMSNorm (standard formulation)",
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

# MoE-specific patches reused verbatim from the GPU config.
config.replace_class(
    "Qwen3VLMoeTextExperts",
    replacement=PatchedQwen3VLMoeTextExperts,
    description="Drop @use_experts_implementation decorator and add VeOmni fused MoE dispatch path",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.forward",
    replacement=qwen3_vl_moe_for_conditional_generation_forward_patched,
    description="Use VeOmni fused loss_function and MoE aux_loss path",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.get_parallel_plan",
    replacement=qwen3_vl_moe_get_parallel_plan_patched,
    description="Register Qwen3VLMoe expert parallel plan for v5 generated modeling",
)


# ================================================================
# Patch: apply_rotary_pos_emb -> NPU fused npu_rotary_mul
# 1. text-side RoPE kernel; identical signature to the upstream
#    `apply_rotary_pos_emb`, internally uses `torch_npu.npu_rotary_mul`
# ================================================================
@config.replace_function(
    "apply_rotary_pos_emb",
    description="NPU fused rotary pos emb (torch_npu.npu_rotary_mul)",
)
def apply_rotary_pos_emb_npu_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # --- Patch.1 ---
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)  # noqa: F821 imported via add_import
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)  # noqa: F821 imported via add_import
    return q_embed.to(q.dtype), k_embed.to(k.dtype)
    # --- Patch.1 ---


# ================================================================
# Patch: apply_rotary_pos_emb_vision -> NPU fused npu_rotary_mul
# 1. vision-side RoPE kernel; reshapes to 4D before the NPU call to
#    satisfy the kernel's rank expectation
# ================================================================
@config.replace_function(
    "apply_rotary_pos_emb_vision",
    description="NPU fused vision rotary pos emb (torch_npu.npu_rotary_mul with 4D reshape)",
)
def apply_rotary_pos_emb_vision_npu_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # --- Patch.1 ---
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q_4d = q.unsqueeze(0).float().contiguous()
    k_4d = k.unsqueeze(0).float().contiguous()
    cos_4d = cos.unsqueeze(0).unsqueeze(2).float()
    sin_4d = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed_4d = torch_npu.npu_rotary_mul(q_4d, cos_4d, sin_4d)  # noqa: F821 imported via add_import
    k_embed_4d = torch_npu.npu_rotary_mul(k_4d, cos_4d, sin_4d)  # noqa: F821 imported via add_import
    q_embed = q_embed_4d.squeeze(0).to(orig_q_dtype).reshape(orig_q_shape)
    k_embed = k_embed_4d.squeeze(0).to(orig_k_dtype).reshape(orig_k_shape)
    return q_embed, k_embed
    # --- Patch.1 ---
