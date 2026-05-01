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
Patch configuration for Qwen3_5Moe NPU/SP patched modeling generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_5_moe.qwen3_5_moe_npu_patch_gen_config -o veomni/models/transformers/qwen3_5_moe/generated --diff

Patches applied:
1. Fused MoE expert replacement (merged gate_up_proj layout).
2. Device-agnostic GatedDeltaNet init and varlen FLA forward.
3. DecoderLayer forward with cu_seq_lens_q passthrough.
4. Fused loss + aux_loss in ForConditionalGeneration.
"""

from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    qwen3_5_gated_deltanet_get_local_conv1d_weight,
    qwen3_5_gated_deltanet_init_patched,
    qwen3_5_model_get_image_features,
    qwen3_5_model_get_placeholder_mask,
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_fast_pos_embed_interpolate,
    qwen3_5_vision_model_forward,
)
from veomni.models.transformers.qwen3_5.qwen3_5_npu_patch_gen_config import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    qwen3_5_gated_deltanet_forward_patched,
    qwen3_5_rmsnorm_forward_patched,
    qwen3_5_rmsnorm_gated_forward_patched,
)
from veomni.models.transformers.qwen3_5_moe.qwen3_5_moe_gpu_patch_gen_config import (
    PatchedQwen3_5MoeExperts,
    qwen3_5_moe_decoder_layer_forward_patched,
    qwen3_5_moe_forcausallm_forward_patched,
    qwen3_5_moe_forconditional_generation_forward_patched,
    qwen3_5_moe_forconditional_generation_get_position_id_func,
    qwen3_5_moe_get_parallel_plan_patched,
    qwen3_5_moe_model_forward_patched,
    qwen3_5_moe_model_init_patched,
    qwen3_5_moe_sparse_moe_block_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    target_file="patched_modeling_qwen3_5_moe_npu.py",
    description="Qwen3_5Moe with mojo_opset NPU replacements, fused MoE, and VeOmni SP/fused loss patches",
)

config.add_import("copy", names=["copy"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch_npu", names=["torch_npu"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_heads_scatter_seq"],
)
config.add_import("veomni.distributed.sequence_parallel", names=["sp_pad_and_slice"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
# Surface ``CausalLMOutputWithLogProbs`` so the patched ``forward`` (re-used
# from the GPU config) can return per-token log-probs in the unified output
# dataclass via dynamic attribute set.
config.add_import("veomni.utils.model_outputs", names=["CausalLMOutputWithLogProbs"])  # noqa: F401
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # TODO: Add torch npu ops chunk_gated_delta_rule and causal_conv1d_fn in the future.
    chunk_gated_delta_rule = None
    causal_conv1d_fn = None
    FusedRMSNormGated = None
    fused_recurrent_gated_delta_rule = None
    causal_conv1d_update = None
    """
)
config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # These are bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    """
)

# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None


config.override_method(
    "Qwen3_5MoeRMSNorm.forward",
    replacement=qwen3_5_rmsnorm_forward_patched,
    description="Use fused rmsnorm to impl zero-centered rmsnorm (1+weight centered formulation)",
)

config.override_method(
    "Qwen3_5MoeRMSNormGated.forward",
    replacement=qwen3_5_rmsnorm_gated_forward_patched,
    description="Use fused rmsnorm and fused swiglu to impl gated rmsnorm",
)

config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb,
    description="Use fused rope to impl partial rotary postion embedding",
)

config.replace_function(
    "apply_rotary_pos_emb_vision",
    replacement=apply_rotary_pos_emb_vision,
    description="Use fused rope to impl rotary postion embedding in vit",
)

# ── Propagate _moe_implementation from top-level config to text_config ────────


config.override_method(
    "Qwen3_5MoeModel.__init__",
    replacement=qwen3_5_moe_model_init_patched,
    description="Propagate _moe_implementation from top-level config to text_config",
)


# ── SparseMoeBlock forward (avoid in-place op on autograd Function output) ────


config.override_method(
    "Qwen3_5MoeSparseMoeBlock.forward",
    replacement=qwen3_5_moe_sparse_moe_block_forward_patched,
    description="Avoid in-place += on custom autograd Function output",
)


# ── ViT patches ───────────────────────────────────────────────────────────────

config.override_method(
    "Qwen3_5MoeModel.get_image_features",
    replacement=qwen3_5_model_get_image_features,
    description="Remove unnecessary split operation to maintain contiguous memory layout.",
)

config.override_method(
    "Qwen3_5MoeModel.get_placeholder_mask",
    replacement=qwen3_5_model_get_placeholder_mask,
    description="Extract multimodal placeholder masks from input_ids using self-defined placeholder IDs.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_5_vision_model_fast_pos_embed_interpolate,
    description="Optimized bilinear interpolation for high-resolution vision embeddings, adapted from vLLM.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.forward",
    replacement=qwen3_5_vision_model_forward,
    description="Optimized vision forward with Sequence Parallel (SP) support and padded cu_seqlens.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.dummy_forward",
    replacement=qwen3_5_vision_model_dummy_forward,
    description="Add dummy_forward to prevent FSDP reduce-scatter hang on uneven multimodal batches.",
)


config.override_method(
    "Qwen3_5MoeModel.forward",
    replacement=qwen3_5_moe_model_forward_patched,
    description=(
        "Optimized multimodal forward supporting Ulysses SP (multimodal scattering), "
        "FSDP-safe dummy vision processing, position_ids shape alignment, and "
        "CPU-GPU sync avoidance via pre-computed metadata."
    ),
)


config.add_post_import_block("""
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}
""")


config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_position_id_func",
    replacement=qwen3_5_moe_forconditional_generation_get_position_id_func,
    description="Expose get_position_id_func to pre-computes position IDs per sample during data preprocessing in worker processes.",
)


# ── MoE Expert replacement (merged gate_up_proj layout) ─────────────────────────


config.replace_class(
    "Qwen3_5MoeExperts",
    replacement=PatchedQwen3_5MoeExperts,
    description="Remove @use_experts_implementation decorator and add VeOmni fused MoE dispatch path",
)


# ── GatedDeltaNet patches (shared with qwen3_5 via name_map) ─────────────────

_NAME_MAP = {"Qwen3_5": "Qwen3_5Moe"}

config.override_method(
    "Qwen3_5MoeGatedDeltaNet.__init__",
    replacement=qwen3_5_gated_deltanet_init_patched,
    name_map=_NAME_MAP,
    description="Use device-agnostic get_device_id() for FusedRMSNormGated init",
)

config.override_method(
    "Qwen3_5MoeGatedDeltaNet._get_local_conv1d_weight",
    replacement=qwen3_5_gated_deltanet_get_local_conv1d_weight,
    name_map=_NAME_MAP,
    description="Shard depthwise conv1d weights for local heads under Ulysses SP",
)

config.override_method(
    "Qwen3_5MoeGatedDeltaNet.forward",
    replacement=qwen3_5_gated_deltanet_forward_patched,
    name_map=_NAME_MAP,
    description="Support varlen flash linear attention and Ulysses SP in Qwen3_5MoeGatedDeltaNet.forward",
)


# ── DecoderLayer forward ────────────────────────────────────────────────────────


config.override_method(
    "Qwen3_5MoeDecoderLayer.forward",
    replacement=qwen3_5_moe_decoder_layer_forward_patched,
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5MoeDecoderLayer.forward",
)


# ── ForCausalLM forward (fused loss + aux_loss) ──────────────────────────────────


config.override_method(
    "Qwen3_5MoeForCausalLM.forward",
    replacement=qwen3_5_moe_forcausallm_forward_patched,
    description="Support fused cross entropy path in Qwen3_5MoeForCausalLM.forward",
)


# ── ForConditionalGeneration forward (fused loss + aux_loss) ─────────────────────


config.override_method(
    "Qwen3_5MoeForConditionalGeneration.forward",
    replacement=qwen3_5_moe_forconditional_generation_forward_patched,
    description="Support fused cross entropy path in Qwen3_5MoeForConditionalGeneration.forward",
)


# ── Expert parallel plan ─────────────────────────────────────────────────────


config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_parallel_plan",
    replacement=qwen3_5_moe_get_parallel_plan_patched,
    description="Register Qwen3_5Moe expert parallel plan for v5 generated modeling",
)
