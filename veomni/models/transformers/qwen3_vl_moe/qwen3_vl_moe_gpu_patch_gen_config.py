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
Patch configuration for Qwen3-VL-MoE transformers>=5.2.0 code generation.

Reuses the full set of qwen3_vl VLM patches via `name_map={"Qwen3VL": "Qwen3VLMoe"}`
(vision SP, deepstack, async Ulysses attention, precomputed position-ids, fused
loss) and layers the MoE-specific patches on top:
  - `Qwen3VLMoeTextExperts` fused-MoE dispatch (gate_up_proj fused path);
  - `Qwen3VLMoeModel.__init__` propagates `_moe_implementation` to `text_config`;
  - `Qwen3VLMoeForConditionalGeneration.{forward, get_parallel_plan}` to route
    through fused loss + aux_loss and register the expert parallel plan.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_vl_moe.qwen3_vl_moe_gpu_patch_gen_config -o veomni/models/transformers/qwen3_vl_moe/generated --diff
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeVisionModel,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
)
from veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    config as qwen3_vl_config,
)
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
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
    target_file="patched_modeling_qwen3_vl_moe_gpu.py",
    description="Qwen3-VL-MoE with VeOmni v5 compatibility (SP + async Ulysses + deepstack + fused MoE + fused loss)",
)

# Reuse the same post-import block / helpers / imports that the qwen3_vl GPU
# config already injects into its generated file. The shared body of all the
# reused VLM patches depends on these helpers (`rot_pos_ids`,
# `_qwen3_vl_async_ulysses_attention_forward`, `get_position_id`) being
# available at module scope in the generated modeling.
config.additional_imports.extend(qwen3_vl_config.additional_imports)
config.post_import_blocks.extend(qwen3_vl_config.post_import_blocks)
config.helpers.extend(qwen3_vl_config.helpers)

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


# ================================================================
# Reused VLM patches from qwen3_vl (name_map rewrites Qwen3VL* -> Qwen3VLMoe*
# inside the patch bodies so they target the sibling classes).
# ================================================================
_NAME_MAP = {"Qwen3VL": "Qwen3VLMoe"}

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
    "Qwen3VLMoeForConditionalGeneration.get_position_id_func",
    replacement=qwen3_vl_get_position_id_func_patched,
    name_map=_NAME_MAP,
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)


# ================================================================
# Patch: Qwen3VLMoeModel.__init__
# 1. propagate `_moe_implementation` from the top-level `Qwen3VLMoeConfig`
#    down to `config.text_config` so `Qwen3VLMoeTextSparseMoeBlock.experts`
#    picks up the correct mode (v5 upstream does not propagate it — the
#    inner MoE classes are built from `config.text_config`)
# ================================================================
@config.override_method(
    "Qwen3VLMoeModel.__init__",
    description="Propagate _moe_implementation from top-level config to text_config",
)
def qwen3_vl_moe_model_init_patched(self, config):
    # --- Patch.1 ---
    moe_implementation = getattr(config, "_moe_implementation", "eager")
    config.text_config._moe_implementation = moe_implementation
    # --- Patch.1 ---

    super().__init__(config)
    self.visual = Qwen3VLMoeVisionModel._from_config(config.vision_config)
    self.language_model = Qwen3VLMoeTextModel._from_config(config.text_config)
    self.rope_deltas = None  # cache rope_deltas here

    # Initialize weights and apply final processing
    self.post_init()


# ================================================================
# Patch: Qwen3VLMoeTextExperts
# 1. drop the upstream `@use_experts_implementation` decorator — routing
#    through `ALL_EXPERTS_FUNCTIONS` bypasses our fused kernel
# 2. add VeOmni fused MoE dispatch via `_moe_implementation` config flag;
#    pass `gate_up_proj` directly as `fc1_1_2_weight` (the v5 modeling
#    already stores it in the `[E, 2*I, H]` fused layout, so no chunk +
#    contiguous overhead is needed)
# ================================================================
@config.replace_class(
    "Qwen3VLMoeTextExperts",
    description="Drop @use_experts_implementation decorator and add VeOmni fused MoE dispatch path",
)
class PatchedQwen3VLMoeTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Replaces the HF class to remove the `@use_experts_implementation` decorator
    (which routes to grouped_mm and bypasses our fused MoE path) and to add
    VeOmni fused MoE dispatch via the `_moe_implementation` config flag.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        # --- Patch.2 ---
        # TODO(kernel-registry): migrate to OpSlot("moe_experts", …) like
        # qwen3_5_moe; reading config at __init__ time forces auto.py to run
        # apply_moe_patch_transformers_v4 *before* loader.load_model.
        self._moe_implementation = getattr(config, "_moe_implementation", "eager")
        # --- Patch.2 ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Modification: OpSlot guard — dispatch to fused MoE kernel when bound.
        if veomni_moe_experts_forward.use_non_eager_impl:
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
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ================================================================
# Patch: Qwen3VLMoeModel.forward
# MoE-specific clone of the dense qwen3_vl model forward. The shared
# body (SP + precomputed position-id + dummy-forward + deepstack) is
# identical, but the return type is `Qwen3VLMoeModelOutputWithPast`
# which carries an extra `router_logits` field — dropping it on the
# return statement would silence the MoE load-balancing loss (router
# collapse) since `Qwen3VLMoeForConditionalGeneration.forward` reads
# `outputs.router_logits`.
# ================================================================
@config.override_method(
    "Qwen3VLMoeModel.forward",
    description="VeOmni SP + precomputed position-id + dummy-forward + deepstack; preserve MoE router_logits",
)
def qwen3_vl_moe_model_forward_patched(
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
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3VLMoeModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    if video_mask is None and image_mask is None:
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids_full = torch.cat(input_ids_list, dim=1)
        else:
            input_ids_full = input_ids
        image_mask, video_mask = self.get_placeholder_mask(input_ids_full)
    # --- Patch.2 ---

    # --- Patch.3 ---
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)
    # --- Patch.3 ---

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    fake_deepstack = None

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
            deepstack_image_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_image_embeds
            ]
        # --- Patch.1 ---

        n_image_tokens = image_mask.sum().long().item()
        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        image_embeds = image_embeds[:n_image_tokens]
        deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            seq_len = image_mask.shape[1]
            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            deepstack_offset = image_mask[:, :rank_start].sum().item()
            image_mask = image_mask[:, rank_start:rank_end]
            deepstack_len = image_mask.sum().item()

            deepstack_image_embeds = [
                embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_image_embeds
            ]
        # --- Patch.1 ---

    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_vision = self.visual.dummy_forward()
        fake_embeds = fake_vision.pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        fake_deepstack = fake_vision.deepstack_features
        # --- Patch.4 ---

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithDeepstackFeatures = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output
        deepstack_video_embeds = video_outputs.deepstack_features

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
            deepstack_video_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_video_embeds
            ]
        # --- Patch.1 ---

        n_video_tokens = video_mask.sum().long().item()
        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        video_embeds = video_embeds[:n_video_tokens]
        deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            seq_len = video_mask.shape[1]
            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            deepstack_offset = video_mask[:, :rank_start].sum().item()
            video_mask = video_mask[:, rank_start:rank_end]
            deepstack_len = video_mask.sum().item()

            deepstack_video_embeds = [
                embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_video_embeds
            ]
        # --- Patch.1 ---

    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_vision = self.visual.dummy_forward()
        fake_embeds = fake_vision.pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        fake_deepstack = fake_vision.deepstack_features
        # --- Patch.4 ---

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    visual_pos_masks = None
    deepstack_visual_embeds = None

    if pixel_values is not None and pixel_values_videos is not None:
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif pixel_values is not None:
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif pixel_values_videos is not None:
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds
    else:
        # --- Patch.4 ---
        if fake_deepstack is not None:
            deepstack_visual_embeds = fake_deepstack
        # --- Patch.4 ---

    if position_ids is None:
        # --- Patch.5 ---
        if isinstance(attention_mask, dict):
            attention_mask_tensor = attention_mask.get("full_attention", None)
        else:
            attention_mask_tensor = attention_mask
        if get_parallel_state().sp_enabled:
            raise RuntimeError(
                "Qwen3VLMoeModel.forward: position_ids is None while sequence parallel "
                "is enabled; multimodal position_ids must be precomputed via "
                "`get_position_id_func` in the VeOmni data pipeline."
            )
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_tensor,
            past_key_values=past_key_values,
        )
        # --- Patch.5 ---
    else:
        # --- Patch.5 ---
        if position_ids.dim() == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()
        # --- Patch.5 ---

    # --- Patch.3 ---
    kwargs.update(flash_attn_kwargs)
    # --- Patch.3 ---

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLMoeModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=getattr(outputs, "router_logits", None),
        rope_deltas=self.rope_deltas,
    )


# ================================================================
# Patch: Qwen3VLMoeForConditionalGeneration.forward
# 1. use the unified VeOmni fused loss_function path — avoids
#    materializing full-vocab logits when labels is provided
# 2. compute MoE aux_loss via upstream `load_balancing_loss_func` when
#    `output_router_logits=True`; read config from `config.text_config`
#    since the VLM top-level wraps a nested text config
# ================================================================
@config.override_method(
    "Qwen3VLMoeForConditionalGeneration.forward",
    description="Use VeOmni fused loss_function and MoE aux_loss path",
)
def qwen3_vl_moe_for_conditional_generation_forward_patched(
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
) -> tuple | Qwen3VLMoeCausalLMOutputWithPast:
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
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            # Modification: VeOmni's patched `loss_function` (via LOSS_MAPPING)
            # returns (loss, logits); unpack to match the OpSlot branch above.
            loss, logits = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    # --- Patch.2 ---
    aux_loss = None
    if kwargs.get("output_router_logits", False):
        # Modification: OpSlot guard for load-balancing loss.
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)
    # --- Patch.2 ---

    return Qwen3VLMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=getattr(outputs, "router_logits", None),
    )


# ================================================================
# Patch: Qwen3VLMoeForConditionalGeneration.get_parallel_plan
# 1. register the expert parallel plan on the v5 generated modeling so
#    `.mlp.experts.gate_up_proj` / `.down_proj` get `Shard(0)` under EP
# ================================================================
@config.override_method(
    "Qwen3VLMoeForConditionalGeneration.get_parallel_plan",
    description="Register Qwen3VLMoe expert parallel plan for v5 generated modeling",
)
def qwen3_vl_moe_get_parallel_plan_patched(self):
    # --- Patch.1 ---
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
    # --- Patch.1 ---
