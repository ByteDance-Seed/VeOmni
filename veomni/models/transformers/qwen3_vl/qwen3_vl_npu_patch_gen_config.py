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

import torch

from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    qwen3_vl_for_conditional_generation_forward_patched,
    qwen3_vl_get_position_id_func_patched,
    qwen3_vl_model_forward_patched,
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
config.add_import("torch_npu", is_from_import=False)


# ================================================================
# Shared GPU patches (SP / deepstack / fused-CE / async Ulysses / ...)
# ================================================================
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
    "Qwen3VLForConditionalGeneration.forward",
    replacement=qwen3_vl_for_conditional_generation_forward_patched,
    description="Use VeOmni unified fused loss_function path",
)


# ================================================================
# Patch: apply_rotary_pos_emb -> NPU fused npu_rotary_mul
# 1. the text-side RoPE kernel; identical signature to the upstream
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
# 1. the vision-side RoPE kernel; reshapes to 4D before the NPU call
#    to satisfy the kernel's rank expectation
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


# ================================================================
# Patch: Qwen3VLTextRMSNorm.forward -> NPU fused npu_rms_norm
# 1. swap the full-fp32 variance path for `torch_npu.npu_rms_norm`
#    which stays in the weight dtype and is significantly faster on NPU
# ================================================================
@config.override_method(
    "Qwen3VLTextRMSNorm.forward",
    description="NPU fused RMSNorm (torch_npu.npu_rms_norm)",
)
def qwen3_vl_text_rmsnorm_forward_npu_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # --- Patch.1 ---
    if hidden_states.dtype != self.weight.dtype:
        hidden_states = hidden_states.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]  # noqa: F821 imported via add_import
    # --- Patch.1 ---
