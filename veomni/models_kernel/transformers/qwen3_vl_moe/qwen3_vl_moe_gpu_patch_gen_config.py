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
Patch configuration for Qwen3-VL-MoE VeomniKernel replacements.

Reuses the qwen3_vl VLM patches via `name_map={"Qwen3VL": "Qwen3VLMoe"}`
and adds MoE-specific VeomniKernel calls on top.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3_vl_moe.qwen3_vl_moe_gpu_patch_gen_config -o veomni/models_kernel/transformers/qwen3_vl_moe/generated --diff
"""

from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLMoeModel,
    Qwen3VLMoeModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
)
from veomni.kernels import VeomniKernel
from veomni.models_kernel.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    apply_rotary_pos_emb_patched,
    apply_rotary_pos_emb_vision_patched,
    qwen3_vl_get_metadata_collate_func_patched,
    qwen3_vl_get_position_id_func_patched,
    qwen3_vl_model_get_image_features_patched,
    qwen3_vl_model_get_placeholder_mask_patched,
    qwen3_vl_rmsnorm_forward_patched,
    qwen3_vl_rmsnorm_init_patched,
    qwen3_vl_text_attention_forward_patched,
    qwen3_vl_text_deepstack_process_patched,
    qwen3_vl_vision_attention_forward_patched,
    qwen3_vl_vision_block_forward_patched,
    qwen3_vl_vision_dummy_forward_patched,
    qwen3_vl_vision_fast_pos_embed_interpolate_patched,
    qwen3_vl_vision_forward_patched,
    qwen3_vl_vision_rot_pos_emb_patched,
)
from veomni.models_kernel.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config import (
    config as qwen3_vl_config,
)
from veomni.models_kernel.utils.kernel_utils import empty_bias, resolve_kernel_impl, resolve_moe_impl
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import Qwen3VLMoeCausalLMOutputWithLogProbs


config = PatchConfig(
    source_module="transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
    target_file="patched_modeling_qwen3_vl_moe_gpu.py",
    description="Qwen3-VL-MoE with VeOmni v5 patches and VeomniKernel replacements",
)

# Reuse the same post-import block / helpers / imports that the qwen3_vl GPU
# config already injects into its generated file. The shared body of all the
# reused VLM patches depends on these helpers (`rot_pos_ids`,
# `_qwen3_vl_async_ulysses_attention_forward`, `get_position_id`) being
# available at module scope in the generated modeling.
#
# We deliberately filter out `_Qwen3VLFakeForPosID`: helpers are emitted
# verbatim by patchgen and bypass the per-patch `name_map`, so the qwen3_vl
# class would land here with its body still referencing `Qwen3VLModel`
# (undefined in this generated file). A Moe-specific helper is registered
# below via `@config.add_helper`, mirroring qwen3_vl's helper but binding
# to `Qwen3VLMoeModel`.
config.additional_imports.extend(qwen3_vl_config.additional_imports)
config.post_import_blocks.extend(qwen3_vl_config.post_import_blocks)
config.helpers.extend(h for h in qwen3_vl_config.helpers if h.__name__ != "_Qwen3VLFakeForPosID")

# Surface ``Qwen3VLMoeCausalLMOutputWithLogProbs`` so the patched multimodal
# ``forward`` can return per-token log-probs / entropy as constructor fields
# while preserving ``aux_loss`` and ``rope_deltas``. Mutating
# ``output.log_probs`` / ``output.entropy`` after the base-class constructor
# would bypass ``ModelOutput`` pytree flattening, breaking FSDP2's pre-backward
# unshard hook on ``lm_head`` and triggering ``setStorage … storage of size 0``
# in ``chunk_logprobs.backward`` (parallels VeOmni #731's qwen3_5_moe fix).
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "Qwen3VLMoeCausalLMOutputWithLogProbs"],
)
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["attention_kernel", "empty_bias", "resolve_kernel_impl", "resolve_moe_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
)
config.drop_import_names("Qwen3VLMoeCausalLMOutputWithPast")


# ================================================================
# Helper: _Qwen3VLMoeFakeForPosID  (emitted into the generated file via
# add_helper). Picklable fake `self` used by `get_position_id_func`.
# Mirrors qwen3_vl's `_Qwen3VLFakeForPosID` but binds to `Qwen3VLMoeModel`
# — patchgen emits helpers verbatim (bypassing the per-patch `name_map`),
# so a Moe-specific version is required here.
# ================================================================
@config.add_helper
class _Qwen3VLMoeFakeForPosID(SimpleNamespace):  # noqa: F821 SimpleNamespace declared in qwen3_vl's add_post_import_block which we inherit above
    """Picklable fake `self` used by `get_position_id_func` — must be a
    module-level class so `get_vision_position_ids` survives the pickle
    round-trip that happens when this object reaches a dataloader worker
    via `multiprocessing.spawn`. Assigning a bound method to a plain
    `SimpleNamespace` instance deadlocks on unpickle (`AttributeError:
    'types.SimpleNamespace' object has no attribute 'get_vision_position_ids'`)
    because pickle reduces a bound method to `getattr(self.__self__, name)`
    and `name` is the very attribute being restored from `__dict__`. As a
    class attribute, the method is resolved via class lookup and never has
    to be pickled."""

    def get_vision_position_ids(self, *args, **kwargs):
        return Qwen3VLMoeModel.get_vision_position_ids(self, *args, **kwargs)  # noqa: F821 defined in generated modeling file


# ================================================================
# Reused VLM patches from qwen3_vl (name_map rewrites Qwen3VL* -> Qwen3VLMoe*
# inside the patch bodies so they target the sibling classes).
# ================================================================
_NAME_MAP = {"Qwen3VL": "Qwen3VLMoe"}
config.override_method(
    "Qwen3VLMoeTextRMSNorm.__init__",
    replacement=qwen3_vl_rmsnorm_init_patched,
    name_map=_NAME_MAP,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3VLMoeTextRMSNorm.forward",
    replacement=qwen3_vl_rmsnorm_forward_patched,
    name_map=_NAME_MAP,
    description="Always call the local rms_norm VeomniKernel",
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
    "Qwen3VLMoeForConditionalGeneration.get_position_id_func",
    replacement=qwen3_vl_get_position_id_func_patched,
    name_map=_NAME_MAP,
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)
config.override_method(
    "Qwen3VLMoeForConditionalGeneration.get_metadata_collate_func",
    replacement=qwen3_vl_get_metadata_collate_func_patched,
    name_map=_NAME_MAP,
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope full VeomniKernel",
)
config.replace_function(
    "apply_rotary_pos_emb_vision",
    replacement=apply_rotary_pos_emb_vision_patched,
    description="Always call rope_vision full VeomniKernel",
)


# ================================================================
# Patch: Qwen3VLMoeTextExperts
# 1. drop the upstream `@use_experts_implementation` decorator
# 2. always call a local moe_experts VeomniKernel; pass `gate_up_proj`
#    as merged `fc1_1_2_weight` (v5 already stores `[E, 2*I, H]`)
# ================================================================
@config.replace_class(
    "Qwen3VLMoeTextExperts",
    description="Drop @use_experts_implementation and always call moe_experts VeomniKernel",
)
class PatchedQwen3VLMoeTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Replaces the HF class to remove the `@use_experts_implementation`
    decorator and to call a local ``moe_experts`` VeomniKernel.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.veomni_moe = VeomniKernel("moe_experts", "standard", resolve_moe_impl())

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        unused = empty_bias(self.gate_up_proj)
        return self.veomni_moe(
            hidden_states,
            top_k_weights,
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
        )


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
    r"""
    cache_position (`torch.LongTensor`, *optional*):
        Indices describing the positions of the input sequence tokens in the cache.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    # v5 multimodal RoPE input; consumed here so it is not forwarded to the
    # language model. Derived from input_ids below when not supplied.
    mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
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
        inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=get_parallel_state().sp_group)
    # --- Patch.1 ---

    # --- Patch.6 ---
    # Mirror of qwen3_vl: unpack per-modality ViT kwargs from
    # `multimodal_metadata` (collator-precomputed) so the patched ViT
    # forward can skip the in-forward .tolist() / cu_seqlens build.
    # See .agents/knowledge/multimodal_metadata.md.
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    image_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_image_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_image_max_seqlen"),
        }
    }
    video_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("video_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_video_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_video_max_seqlen"),
        }
    }
    # --- Patch.6 ---

    fake_deepstack = None

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True, **image_vit_kwargs
        )
        image_embeds = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            deepstack_image_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_image_embeds
            ]
        # --- Patch.1 ---

        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        # `masked_scatter` consumes exactly `image_mask.sum()` leading rows; data collator pads
        # vision sequence only at the end. No `image_embeds[:n]` slice needed → no
        # `image_mask.sum().item()` host-device sync. Same for the deepstack list.
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
            pixel_values_videos, video_grid_thw, return_dict=True, **video_vit_kwargs
        )
        video_embeds = video_outputs.pooler_output
        deepstack_video_embeds = video_outputs.deepstack_features

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            video_embeds = gather_outputs(video_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            deepstack_video_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_video_embeds
            ]
        # --- Patch.1 ---

        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        # Same as image branch above: drop the `[:n_video_tokens]` slice + the
        # `.item()` sync.
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
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=get_parallel_state().sp_group)

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
        # v5 `compute_3d_position_ids` gates M-RoPE on `mm_token_type_ids`
        # via its `can_compute_mrope` check; without it the multimodal
        # branch silently falls through and the call returns `None`,
        # leaving `position_ids=None` for the language model. Derive
        # `mm_token_type_ids` from `input_ids` here so the M-RoPE branch
        # actually runs whenever multimodal grids are present.
        if (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            mm_token_type_ids = mm_token_type_ids_from_input_ids(  # noqa: F821 defined via add_helper
                input_ids, self.config
            )
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_tensor,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
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
# Patch: Qwen3VLMoeForConditionalGeneration.__init__
# Bind ForCausalLMLoss + a local load_balancing_loss VeomniKernel.
# ================================================================
@config.override_method(
    "Qwen3VLMoeForConditionalGeneration.__init__",
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
def qwen3_vl_moe_for_conditional_generation_init_patched(self, config):
    super().__init__(config)
    self.model = Qwen3VLMoeModel(config)
    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.veomni_lb = VeomniKernel(
        "load_balancing_loss",
        "standard",
        resolve_kernel_impl("load_balancing_loss_implementation"),
    )
    self.post_init()


# ================================================================
# Patch: Qwen3VLMoeForConditionalGeneration.forward
# 1. always call self.loss_function (ForCausalLMLoss + VeomniKernel)
# 2. aux_loss via local load_balancing_loss kernel after cat to [N, E]
# ================================================================
@config.override_method(
    "Qwen3VLMoeForConditionalGeneration.forward",
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
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
) -> tuple | Qwen3VLMoeCausalLMOutputWithLogProbs:
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
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    # --- Patch.2 ---
    aux_loss = None
    if kwargs.get("output_router_logits", False):
        router_logits = outputs.router_logits
        if router_logits is None or not isinstance(router_logits, tuple):
            aux_loss = 0
        else:
            gate = torch.cat([layer.reshape(-1, layer.shape[-1]) for layer in router_logits], dim=0)
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else gate.new_empty(0)
            aux_loss = self.veomni_lb(gate, mask, top_k=self.config.text_config.num_experts_per_tok)
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)
    # --- Patch.2 ---

    return Qwen3VLMoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=getattr(outputs, "router_logits", None),
        fused_linear_aux=fused_linear_aux,
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
