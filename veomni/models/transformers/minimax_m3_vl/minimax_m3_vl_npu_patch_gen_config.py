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

This path reuses the GPU patch set unchanged. Device-specific behavior (fused
MoE, RMSNorm and sparse attention backends) is selected at runtime through the
op registry, so the NPU artifact differs from the GPU one only in helper order.

Regen command:
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config -o veomni/models/transformers/minimax_m3_vl/generated --diff
"""

from veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config import (
    MiniMaxM3VLCausalLMOutputWithLogProbs,
    PatchedMiniMaxM3VLExperts,
    PatchedMiniMaxM3VLPreTrainedModel,
    _grid_thw_to_list,
    _minimax_m3_indexer_project,
    _minimax_m3_sparse_attention,
    _require_varlen_attention,
    collate_multimodal_metadata,
    minimax_m3_vl_3d_rotary_embedding_forward_patched,
    minimax_m3_vl_attention_bind_ops,
    minimax_m3_vl_attention_forward_patched,
    minimax_m3_vl_get_metadata_collate_func_patched,
    minimax_m3_vl_get_parallel_plan_patched,
    minimax_m3_vl_get_position_id_func_patched,
    minimax_m3_vl_indexer_init_patched,
    minimax_m3_vl_model_forward_patched,
    minimax_m3_vl_rmsnorm_forward_patched,
    minimax_m3_vl_rmsnorm_init_patched,
    minimax_m3_vl_sparse_for_conditional_generation_forward_patched,
    minimax_m3_vl_sparse_for_conditional_generation_init_patched,
    minimax_m3_vl_text_model_forward_patched,
    minimax_m3_vl_vision_attention_bind_ops,
    minimax_m3_vl_vision_attention_forward_patched,
    minimax_m3_vl_vision_dummy_forward_patched,
    minimax_m3_vl_vision_model_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
    target_file="patched_modeling_minimax_m3_vl_npu.py",
    description="MiniMax M3 VL with VeOmni parallel-plan hooks for NPU runtime selection",
)

config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("functools", names=["partial"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import("veomni.ops.config", names=["resolve_op_impl"])
config.add_import("veomni.models.loss_utils", names=["ForCausalLMLoss"])
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
config.add_import("veomni.models.utils.attention_utils", names=["VARLEN_ATTENTION_TYPES"])
config.add_import("veomni.utils.device", names=["IS_NPU_AVAILABLE"])
config.add_helper(_grid_thw_to_list)
config.add_helper(_minimax_m3_indexer_project)
config.add_helper(_require_varlen_attention)
config.add_helper(_minimax_m3_sparse_attention)
config.add_helper(collate_multimodal_metadata)

config.replace_class(
    "MiniMaxM3VLPreTrainedModel",
    replacement=PatchedMiniMaxM3VLPreTrainedModel,
    description="Allow VeOmni flash-attention implementations for the vision tower and dense language layers",
)

config.replace_class(
    "MiniMaxM3VLExperts",
    replacement=PatchedMiniMaxM3VLExperts,
    description="Remove the HF experts decorator and always call the moe_experts swiglu_oai VeomniOp",
)

config.override_method(
    "MiniMaxM3VLIndexer.__init__",
    replacement=minimax_m3_vl_indexer_init_patched,
    description="Freeze the sparse-attention indexer, which receives no gradient",
)
config.override_method(
    "MiniMaxM3VLRMSNorm.__init__",
    replacement=minimax_m3_vl_rmsnorm_init_patched,
    description="Construct a local rms_norm offset VeomniOp",
)
config.override_method(
    "MiniMaxM3VLRMSNorm.forward",
    replacement=minimax_m3_vl_rmsnorm_forward_patched,
    description="Always call the local rms_norm offset VeomniOp",
)
config.modify_init(
    "MiniMaxM3VLVisionAttention",
    replacement=minimax_m3_vl_vision_attention_bind_ops,
    description="Bind instance-local attention VeomniOp",
)
config.override_method(
    "MiniMaxM3VLVisionAttention.forward",
    replacement=minimax_m3_vl_vision_attention_forward_patched,
    description="Always call the local attention VeomniOp",
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
config.modify_init(
    "MiniMaxM3VLAttention",
    replacement=minimax_m3_vl_attention_bind_ops,
    description="Bind instance-local attention VeomniOps: standard attention for dense layers, MSA for sparse ones",
)
config.override_method(
    "MiniMaxM3VLAttention.forward",
    replacement=minimax_m3_vl_attention_forward_patched,
    description="Run dense layers through varlen attention and sparse layers through the TND sparse attention operators",
)
config.override_method(
    "MiniMaxM3VLTextModel.forward",
    replacement=minimax_m3_vl_text_model_forward_patched,
    description="Accept only packed inputs and reject padded batches and KV cache",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.__init__",
    replacement=minimax_m3_vl_sparse_for_conditional_generation_init_patched,
    description="Bind the ForCausalLMLoss cross-entropy VeomniOp",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.forward",
    replacement=minimax_m3_vl_sparse_for_conditional_generation_forward_patched,
    description="Always call the bound ForCausalLMLoss and unpack its fused-loss tuple",
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
