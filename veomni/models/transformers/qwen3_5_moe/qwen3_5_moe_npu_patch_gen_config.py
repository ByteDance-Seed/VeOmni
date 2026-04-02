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

from copy import copy
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeModel,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.transformers.qwen3_5.qwen3_5_npu_patch_gen_config import (
    qwen3_5_moe_rmsnorm_forward_patched,
    qwen3_5_gated_deltanet_forward_patched,
    qwen3_5_gated_deltanet_get_local_conv1d_weight,
    qwen3_5_gated_deltanet_init_patched,
    qwen3_5_model_get_image_features,
    qwen3_5_model_get_placeholder_mask,
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_fast_pos_embed_interpolate,
    qwen3_5_vision_model_forward,
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX


logger = logging.get_logger(__name__)


config = PatchConfig(
    source_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    target_file="patched_modeling_qwen3_5_moe_npu.py",
    description="Qwen3_5_moe with VeOmni language-model SP and fused loss patches",
)

config.add_import("copy", names=["copy"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
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
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # These are bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_causal_conv1d = OpSlot("causal_conv1d", "standard")
    veomni_chunk_gated_delta_rule = OpSlot("chunk_gated_delta_rule", "standard")
    veomni_cross_entropy_loss = OpSlot("cross_entropy_loss", "standard")

    FusedRMSNormGated = None
    fused_recurrent_gated_delta_rule = None
    causal_conv1d_update = None
    causal_conv1d_fn = None
    chunk_gated_delta_rule = None
    """
)

# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None


# ── RMSNorm (OpSlot guard, functional Liger kernel) ──────────────────────────

config.override_method(
    "Qwen3_5MoeRMSNorm.forward",
    replacement=qwen3_5_moe_rmsnorm_forward_patched,
    description="Use eager Qwen3Next-style RMSNorm (1+weight centered formulation) for NPU patchgen",
)


# NOTE: apply_rotary_pos_emb is NOT replaced with LigerKernel rotary because
# Qwen3_5Moe uses partial_rotary_factor=0.25 with mrope_interleaved=True.
# The HF implementation correctly handles partial rotary (applying RoPE only
# to the first `rotary_dim` dims and passing through the rest), while
# liger_rotary_pos_emb applies RoPE to the full head_dim, producing incorrect
# results and NaN in attention output.


# ── Propagate _moe_implementation from top-level config to text_config ────────


@config.override_method(
    "Qwen3_5MoeModel.__init__",
    description="Propagate _moe_implementation from top-level config to text_config",
)
def qwen3_5_moe_model_init_patched(self, config):
    # Propagate _moe_implementation so SparseMoeBlock picks up the correct mode.
    moe_implementation = getattr(config, "_moe_implementation", "eager")
    config.text_config._moe_implementation = moe_implementation

    super().__init__(config)
    self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
    self.language_model = Qwen3_5MoeTextModel._from_config(config.text_config)
    self.rope_deltas = None  # cache rope_deltas here

    # Initialize weights and apply final processing
    self.post_init()


# ── SparseMoeBlock forward (avoid in-place op on autograd Function output) ────


@config.override_method(
    "Qwen3_5MoeSparseMoeBlock.forward",
    description="Avoid in-place += on custom autograd Function output",
)
def qwen3_5_moe_sparse_moe_block_forward_patched(
    self, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = self.shared_expert(hidden_states_reshaped)
    _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

    # Modification: use out-of-place add instead of `expert_output += shared_expert_output`
    # to avoid "Output of MergedFc1TritonFusedMoeExpertFunctionBackward is a view and is
    # being modified inplace" RuntimeError from PyTorch autograd.
    expert_output = expert_output + shared_expert_output
    expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
    return expert_output


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


@config.override_method(
    "Qwen3_5MoeModel.forward",
    description=(
        "Optimized multimodal forward supporting Ulysses SP (multimodal scattering), "
        "FSDP-safe dummy vision processing, position_ids shape alignment, and "
        "CPU-GPU sync avoidance via pre-computed metadata."
    ),
)
def qwen3_5_moe_model_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5MoeModelOutputWithPast:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Token type IDs for multimodal inputs.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.1: Support Ulysses SP by using pre-computed image and video masks ---
    # We use pre-computed masks to ensure all ranks have a consistent view of multimodal
    # placeholder positions. If masks are not provided, we reconstruct the full sequence
    # via all_gather to compute them locally.
    image_mask = kwargs.get("image_mask", None)
    video_mask = kwargs.get("video_mask", None)

    # if None, calculate mask
    if video_mask is None and image_mask is None:
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for i in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids = torch.cat(input_ids_list, dim=0)
        image_mask, video_mask = self.get_placeholder_mask(input_ids)
    # --- Patch.1 ---

    # --- Patch.4: Pop pre-computed Flash Attention kwargs to avoid ViT forward re-computation ---
    # The LM-level flash-attention kwargs (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`) are injected for packed-sequence attention. They must not reach the ViT, which computes its own `cu_seqlens`
    flash_attn_kwargs = {}
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)
    # --- Patch.4 ---

    # --- Patch.1: Support Ulysses SP by transposing layout for multimodal scattering ---
    if get_parallel_state().sp_enabled:
        # Transpose from (batch, local_seq, full_hidden) to (batch, full_seq, local_hidden).
        # This gives each rank visibility over the ENTIRE sequence length, which is
        # necessary to scatter vision features into their correct global positions
        # as defined by the global pre-computed masks.
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output

        # --- Patch.1: Shard image_embeds for sequence parallel scatter ---
        if get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        n_image_tokens = image_mask.sum().long().item()
        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        # Slice tensor to drop any padded image tokens
        image_embeds = image_embeds[:n_image_tokens]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

        # sequence parallel patch for image_mask
        if get_parallel_state().sp_enabled:
            seq_len = image_mask.shape[1]

            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            image_mask = image_mask[:, rank_start:rank_end]
        # --- Patch.1 ---
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.2: Dummy forward to prevent FSDP reduce-scatter hang on uneven multimodal batches ---
        # add dummy ViT forward to avoid FSDP reduce-scatter hang
        # when some ranks get None pixel_values while others get valid pixel_values
        vision_output = self.visual.dummy_forward()
        fake_embeds = vision_output.pooler_output
        fake_embeds = fake_embeds.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.2 ---

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithPooling = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output

        # --- Patch.1: Shard video_embeds for sequence parallel scatter ---
        # sequence parallel patch for video embeds
        if get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        n_video_tokens = video_mask.sum().long().item()
        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )

        # Slice tensor to drop any padded video tokens
        video_embeds = video_embeds[:n_video_tokens]
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

        # sequence parallel patch for video_mask
        if get_parallel_state().sp_enabled:
            seq_len = video_mask.shape[1]

            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            video_mask = video_mask[:, rank_start:rank_end]
        # --- Patch.1 ---
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.2: Dummy forward for video encoder to avoid FSDP hang ---
        # add dummy ViT forward to avoid FSDP reduce-scatter hang
        # when some ranks get None pixel_values_videos while others get valid pixel_values_videos
        vision_output = self.visual.dummy_forward()
        fake_embeds = vision_output.pooler_output
        fake_embeds = fake_embeds.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.2 ---

    # --- Patch.1: Final transpose back to standard sequence-sharded layout ---
    if get_parallel_state().sp_enabled:
        # Restore the layout to (batch, local_seq, full_hidden) for subsequent
        # transformer layers, which expect standard Sequence Parallel sharding.
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )
    else:
        # --- Patch.3: Transpose pre-computed position_ids if they follow VeOmni collation format ---
        # When position_ids are pre-computed during data preprocessing (for varlen/packed data),
        # they are typically collated into (batch_size, 3, seq_len) shape. We transpose them
        if position_ids.dim() == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()
        # --- Patch.3 ---

    # --- Patch.4: Restore pre-computed Flash Attention kwargs for language model ---
    kwargs.update(flash_attn_kwargs)
    # --- Patch.4 ---

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    return Qwen3_5MoeModelOutputWithPast(
        **outputs,
        rope_deltas=self.rope_deltas,
    )


config.add_post_import_block("""
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}
""")


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_position_id_func",
    description="Expose get_position_id_func to pre-computes position IDs per sample during data preprocessing in worker processes.",
)
def qwen3_5_moe_forconditional_generation_get_position_id_func(self):
    fake_config = copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = SimpleNamespace(config=fake_config)
    return partial(get_position_id, Qwen3_5MoeModel.get_rope_index, fake_model)  # noqa: F821 already defined via above `add_post_import_block`


# ── MoE Expert replacement (merged gate_up_proj layout) ─────────────────────────


@config.replace_class(
    "Qwen3_5MoeExperts",
    description="Remove @use_experts_implementation decorator and add OpSlot-based fused MoE dispatch",
)
class PatchedQwen3_5MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Replaces the HF class to remove the @use_experts_implementation decorator
    (which routes to grouped_mm and bypasses our fused MoE path) and to add
    VeOmni fused MoE dispatch via OpSlot.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Modification: OpSlot guard — dispatch to fused MoE kernel when bound.
        if veomni_moe_experts_forward.has_kernel:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        # Original HF eager loop below, unchanged.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            intermediate_dim = self.intermediate_dim
            gate_w = self.gate_up_proj[expert_idx, :intermediate_dim, :]
            up_w = self.gate_up_proj[expert_idx, intermediate_dim:, :]
            gate = nn.functional.linear(current_state, gate_w)
            up = nn.functional.linear(current_state, up_w)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


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


@config.override_method(
    "Qwen3_5MoeDecoderLayer.forward",
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5MoeDecoderLayer.forward",
)
def qwen3_5_moe_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.FloatTensor:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Modification: read varlen metadata from kwargs and enforce it for linear-attention varlen kernels.
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
    assert cu_seq_lens_q is not None, (
        "cu_seq_lens_q must be provided to support varlen Flash Linear Attention, varlen Conv1D,"
        "and to remove the full Flash Attention CPU-GPU sync."
    )

    # Token Mixer
    if self.layer_type == "linear_attention":
        # Modification: pass cu_seq_lens_q through to Qwen3_5MoeGatedDeltaNet.forward.
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
        )
    elif self.layer_type == "full_attention":
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # For the MoE layers, we need to unpack
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states
    return hidden_states


# ── ForConditionalGeneration forward (fused loss + aux_loss) ─────────────────────


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.forward",
    description="Support fused cross entropy path in Qwen3_5MoeForConditionalGeneration.forward",
)
def qwen3_5_moe_forconditional_generation_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5MoeCausalLMOutputWithPast:
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_cross_entropy_loss.has_kernel:
            loss, logits = veomni_cross_entropy_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if kwargs.get("output_router_logits", False):
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    return Qwen3_5MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=outputs.router_logits,
    )


# ── Expert parallel plan ─────────────────────────────────────────────────────


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_parallel_plan",
    description="Register Qwen3_5Moe expert parallel plan for v5 generated modeling",
)
def qwen3_5_moe_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()