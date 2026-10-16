# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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
Patch configuration for MiniMax M3 VL NPU code generation.

This path mirrors the GPU patch set and binds MiniMax's SwiGLU-OAI experts to
the NPU fused-MoE registry variant while retaining the upstream eager fallback
when eager MoE is selected without expert parallelism.

Regen command:
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config -o veomni/models/transformers/minimax_m3_vl/generated --diff
"""

from veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config import (
    MiniMaxM3VLCausalLMOutputWithLogProbs,
    PatchedMiniMaxM3VLExperts,
    PatchedMiniMaxM3VLPreTrainedModel,
    _grid_thw_to_list,
    _minimax_m3_indexer_project,
    collate_multimodal_metadata,
    minimax_m3_vl_3d_rotary_embedding_forward_patched,
    minimax_m3_vl_attention_forward_patched,
    minimax_m3_vl_get_metadata_collate_func_patched,
    minimax_m3_vl_get_parallel_plan_patched,
    minimax_m3_vl_get_position_id_func_patched,
    minimax_m3_vl_model_forward_patched,
    minimax_m3_vl_rmsnorm_forward_patched,
    minimax_m3_vl_sparse_for_conditional_generation_forward_patched,
    minimax_m3_vl_text_model_forward_patched,
    minimax_m3_vl_vision_dummy_forward_patched,
    minimax_m3_vl_vision_model_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
    target_file="patched_modeling_minimax_m3_vl_npu.py",
    description="MiniMax M3 VL with VeOmni parallel-plan hooks for NPU runtime selection",
)

config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops.dispatch", names=["OpSlot"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["CausalLMOutputWithLogProbs", "FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")
# Only the multimodal model is supported; the text-only causal LM is not generated.
config.exclude_from_output("load_balancing_loss_func", "MiniMaxM3VLForCausalLM")
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "slice_input_tensor", "sp_pad_and_slice"],
)
config.add_import(
    "veomni.ops.kernels.attention.ulysses",
    names=["gather_seq_scatter_kv_heads", "prepare_ulysses_qkv", "restore_ulysses_output"],
)
config.add_import(
    "veomni.ops.kernels.minimax_sparse_attention",
    names=["minimax_sparse_attention_eager", "minimax_sparse_indexer_eager"],
)
config.add_import("veomni.models.transformers.attention_utils", names=["VARLEN_ATTENTION_TYPES"])
config.add_import("veomni.utils.device", names=["IS_NPU_AVAILABLE"])
config.add_post_import_block(
    """
veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
veomni_moe_experts_forward = OpSlot("moe_experts", "swiglu_oai")
veomni_msa_indexer = OpSlot("minimax_sparse_attention", "indexer")
veomni_msa_attention = OpSlot("minimax_sparse_attention", "attention")
"""
)
config.add_helper(_grid_thw_to_list)
config.add_helper(_minimax_m3_indexer_project)
config.add_helper(collate_multimodal_metadata)

config.replace_class(
    "MiniMaxM3VLPreTrainedModel",
    replacement=PatchedMiniMaxM3VLPreTrainedModel,
    description="Allow VeOmni flash-attention implementations for the config-driven vision tower",
)

config.replace_class(
    "MiniMaxM3VLExperts",
    replacement=PatchedMiniMaxM3VLExperts,
    description="Remove the HF experts decorator and add SwiGLU-OAI fused MoE dispatch",
)

config.override_method(
    "MiniMaxM3VLRMSNorm.forward",
    replacement=minimax_m3_vl_rmsnorm_forward_patched,
    description="Use VeOmni's Gemma-style fused RMSNorm backend when selected",
)

config.override_method(
    "MiniMaxM3VL3DRotaryEmbedding.forward",
    replacement=minimax_m3_vl_3d_rotary_embedding_forward_patched,
    description="Consume collator-precomputed MiniMax vision grid lists when available",
)
config.override_method(
    "MiniMaxM3VLVisionModel.forward",
    replacement=minimax_m3_vl_vision_model_forward_patched,
    description="Add MiniMax vision SP RoPE and varlen-attention metadata handling",
)
config.override_method(
    "MiniMaxM3VLVisionModel.dummy_forward",
    replacement=minimax_m3_vl_vision_dummy_forward_patched,
    description="Provide MiniMax dummy vision forward for asymmetric FSDP batches",
)
config.override_method(
    "MiniMaxM3VLModel.forward",
    replacement=minimax_m3_vl_model_forward_patched,
    description="Add MiniMax VLM metadata fast path and FSDP dummy vision branch",
)
config.override_method(
    "MiniMaxM3VLAttention.forward",
    replacement=minimax_m3_vl_attention_forward_patched,
    description="Run packed MiniMax language attention through the TND sparse attention operators",
)
config.override_method(
    "MiniMaxM3VLTextModel.forward",
    replacement=minimax_m3_vl_text_model_forward_patched,
    description="Accept only packed inputs and reject padded batches, KV cache and CP",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.forward",
    replacement=minimax_m3_vl_sparse_for_conditional_generation_forward_patched,
    description="Unpack VeOmni causal LM loss tuple and route fused loss kernels when selected",
)

config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_parallel_plan",
    replacement=minimax_m3_vl_get_parallel_plan_patched,
    description="Register MiniMax M3 VL expert parallel plan for the multimodal training path",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_position_id_func",
    replacement=minimax_m3_vl_get_position_id_func_patched,
    description="Use VeOmni's default 1-D packed-sequence position IDs for MiniMax M3 VL SFT data",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func",
    replacement=minimax_m3_vl_get_metadata_collate_func_patched,
    description="Expose MiniMax CPU-side vision grid metadata derivation to the VeOmni collator",
)

config.add_helper_after("MiniMaxM3VLCausalLMOutputWithPast", MiniMaxM3VLCausalLMOutputWithLogProbs)
