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
source module -- ops-tree eager KDA primitives, eager MLA attention -- are
NPU-safe, and the VeomniOp handles resolve per ops config).
"""

from veomni.models.transformers.kimi_k3.kimi_k3_gpu_patch_gen_config import (
    PatchedKimiK3Experts,
    PatchedKimiK3SparseMoeBlock,
    kimi_decoder_layer_attn_selective_recompute_patched,
    kimi_decoder_layer_call_patched,
    kimi_decoder_layer_forward_attn_residual_patched,
    kimi_decoder_layer_forward_patched,
    kimi_delta_attention_forward_patched,
    kimi_delta_attention_init_patched,
    kimi_k3_for_conditional_generation_forward_patched,
    kimi_k3_for_conditional_generation_init_patched,
    kimi_k3_get_extra_collate_infos_patched,
    kimi_k3_get_parallel_plan_patched,
    kimi_k3_get_position_id_func_patched,
    kimi_k3_merge_input_ids_with_image_features_patched,
    kimi_k3_pretrained_init_weights_patched,
    kimi_k3_tie_weights_patched,
    kimi_linear_for_causal_lm_forward_patched,
    kimi_linear_for_causal_lm_init_patched,
    kimi_linear_get_parallel_plan_patched,
    kimi_linear_model_init_patched,
    kimi_mla_attention_forward_patched,
    kimi_mla_attention_init_patched,
    kimi_moe_gate_forward_patched,
    kimi_pretrained_init_weights_patched,
    kimi_rmsnorm_forward_patched,
    kimi_rmsnorm_init_patched,
    moonvit3d_pretrained_init_weights_patched,
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
# imports). The module-level helpers (DTensor-safe embedding zeroing, the
# head-sharded SP KDA forward) are mirrored too: the re-registered patch
# bodies below resolve those names from the generated module's namespace.
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.drop_imported_names.update(gpu_config.drop_imported_names)
config.helpers.extend(gpu_config.helpers)


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

# ── KDA chunked linear attention (instance-local VeomniOp) ───────────────────


config.override_method(
    "KimiDeltaAttention.__init__",
    replacement=kimi_delta_attention_init_patched,
    description="Define the instance-local chunk_kda VeomniOp declared by the source",
)

config.override_method(
    "KimiDeltaAttention.forward",
    replacement=kimi_delta_attention_forward_patched,
    description=(
        "Sequence-parallel KDA: head-sharded ulysses all-to-all by default "
        "(K3_KDA_HEAD_SHARD=1) or the legacy full-sequence gather (K3_KDA_HEAD_SHARD=0)"
    ),
)

# ── Decoder layer: EP selective recompute (moved out of the vendored source) ──


config.override_method(
    "KimiDecoderLayer._attn_selective_recompute",
    replacement=kimi_decoder_layer_attn_selective_recompute_patched,
    description=(
        "EP selective-recompute policy (VeOmni-only; upstream has no EP): "
        "checkpoint only the collective-free attention stack under EP"
    ),
)

config.override_method(
    "KimiDecoderLayer.__call__",
    replacement=kimi_decoder_layer_call_patched,
    description=(
        "Skip the whole-layer GradientCheckpointingLayer checkpoint under EP "
        "(forward() checkpoints the attention stack itself)"
    ),
)

config.override_method(
    "KimiDecoderLayer.forward",
    replacement=kimi_decoder_layer_forward_patched,
    description=(
        "EP selective recompute at the call site too (moved out of the vendored "
        "source together with the policy method; upstream has no expert "
        "parallelism): checkpoint only the collective-free attention stack under EP"
    ),
)

config.override_method(
    "KimiDecoderLayer._forward_attn_residual",
    replacement=kimi_decoder_layer_forward_attn_residual_patched,
    description=(
        "Attention-residual forward carrying the same EP selective-recompute "
        "call site (see KimiDecoderLayer.forward)"
    ),
)

# ── MLA: instance-local attention VeomniOp + sequence parallel ───────────────────────────────────────────────


config.override_method(
    "KimiMLAAttention.__init__",
    replacement=kimi_mla_attention_init_patched,
    description=(
        "Bind the instance-local attention VeomniOp (veomni_attn) at construction so an "
        "unsupported attention implementation fails here instead of raising from inside "
        "forward (full-body override: patchgen has no init-append mechanism)"
    ),
)

config.override_method(
    "KimiMLAAttention.forward",
    replacement=kimi_mla_attention_forward_patched,
    description=(
        "Sequence-parallel MLA: full-sequence gather/narrow around the attention, or "
        "head-sharded ulysses inside the veomni_* attention adapters"
    ),
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


config.override_method(
    "MoonViT3dPretrainedModel._init_weights",
    replacement=moonvit3d_pretrained_init_weights_patched,
    description=(
        "Vision-tower init under the FSDP2 meta flow: composite-model smart_apply dispatches "
        "sub-models to their own _init_weights and the default chain has no branch for the "
        "learnable 2D pos-emb (or the KDA o_norm / to_empty-zeroed tower norms)"
    ),
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
    "KimiK3ForConditionalGeneration._merge_input_ids_with_image_features",
    replacement=kimi_k3_merge_input_ids_with_image_features_patched,
    description=(
        "Three input forms: pre-expanded media-pad rows (the kimi_k3 data transform; "
        "length-preserving masked_scatter), legacy one-placeholder-per-image rows "
        "(vendored HF occupation-table expansion), and the no-image passthrough"
    ),
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
