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
Patch configuration for Qwen2.5-VL transformers>=5.2.0 code generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen2_5vl.qwen2_5_vl_gpu_patch_gen_config -o veomni/models/transformers/qwen2_5vl/generated --diff
"""

import copy
from functools import partial
from types import SimpleNamespace
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    is_flash_attention_requested,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    pad_tensor,
    sp_pad_and_slice,
    unpad_tensor,
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX


config = PatchConfig(
    source_module="transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    target_file="patched_modeling_qwen2_5_vl_gpu.py",
    description="Qwen2.5-VL with VeOmni v5 compatibility (SP + window attention + fused-loss)",
)

config.add_import("copy", is_from_import=False)
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_heads_scatter_seq", "gather_seq_scatter_heads", "pad_tensor", "sp_pad_and_slice", "unpad_tensor"],
)
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
# Surface ``CausalLMOutputWithLogProbs`` so the patched ``forward`` can return
# per-token log-probs in the unified output dataclass via dynamic attribute set.
config.add_import("veomni.utils.model_outputs", names=["CausalLMOutputWithLogProbs"])  # noqa: F401

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    """
)


# ================================================================
# Patch: Qwen2_5_VLVisionAttention.forward
# 1. accept precomputed max_seqlen from outer forward to avoid
#    per-layer `(cu_seqlens[1:] - cu_seqlens[:-1]).max()` CPU-GPU sync
# ================================================================
@config.override_method(
    "Qwen2_5_VLVisionAttention.forward",
    description="Use precomputed max_seqlen passed from outer forward to avoid per-layer CPU-GPU sync",
)
def qwen2_5_vl_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    # --- Patch.1 ---
    max_seqlen: int,
    # --- Patch.1 ---
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    if is_flash_attention_requested(self.config):
        # --- Patch.1 ---
        # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        # --- Patch.1 ---
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ================================================================
# Patch: Qwen2_5_VLVisionBlock.forward
# 1. thread the precomputed max_seqlen down into the attention call
#    (paired with the Qwen2_5_VLVisionAttention.forward patch above)
# ================================================================
@config.override_method(
    "Qwen2_5_VLVisionBlock.forward",
    description="Propagate precomputed max_seqlen to attention to avoid per-layer CPU-GPU sync",
)
def qwen2_5_vl_vision_block_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    # --- Patch.1 ---
    max_seqlen: int,
    # --- Patch.1 ---
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        # --- Patch.1 ---
        max_seqlen=max_seqlen,
        # --- Patch.1 ---
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


# ================================================================
# Patch: Qwen2_5_VisionTransformerPretrainedModel.forward
# 1. SP all-to-all to get full-seq hidden_states for window attention
#    (gather_seq_scatter_heads / gather_heads_scatter_seq around the
#    window-index permutation + merger fill-back)
# 2. SP-pad cu_seqlens / cu_window_seqlens / position embeddings so the
#    padded tokens participate in a valid attention window
# 3. precompute max_seqlen / win_max_seqlen here (once) to avoid
#    per-layer CPU-GPU sync inside Qwen2_5_VLVisionAttention.forward
# 4. return BaseModelOutputWithPooling (v5 contract: last_hidden_state =
#    pre-merger tokens, pooler_output = post-merger tokens)
# ================================================================
@config.override_method(
    "Qwen2_5_VisionTransformerPretrainedModel.forward",
    description="VeOmni SP + window-attention all-to-all + precomputed max_seqlen",
)
def qwen2_5_vit_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> BaseModelOutputWithPooling:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # --- Patch.1 ---
    unpadded_dim_size = cu_seqlens[-1]
    sp_padding_size = 0
    if get_parallel_state().sp_enabled:
        hidden_states = gather_seq_scatter_heads(
            hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        )
        sp_padding_size = hidden_states.size(0) - unpadded_dim_size
        if sp_padding_size > 0:
            hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
    # --- Patch.1 ---

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

    if get_parallel_state().sp_enabled:
        if sp_padding_size > 0:
            # --- Patch.1 ---
            hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
            # --- Patch.1 ---
            # --- Patch.2 ---
            emb = pad_tensor(emb, dim=0, padding_size=sp_padding_size)
            new_cumsum = cu_seqlens[-1] + sp_padding_size
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
            cu_window_seqlens = torch.cat([cu_window_seqlens, new_cumsum.unsqueeze(0)], dim=0)
            # --- Patch.2 ---
        # --- Patch.1 ---
        hidden_states = gather_heads_scatter_seq(
            hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        )
        # --- Patch.1 ---
        # --- Patch.2 ---
        emb = sp_pad_and_slice(emb, dim=0)
        # --- Patch.2 ---

    position_embeddings = (emb.cos(), emb.sin())

    # --- Patch.3 ---
    win_max_seqlen = (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().detach().cpu().item()
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
    # --- Patch.3 ---

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
            max_seqlens_now = max_seqlen
        else:
            cu_seqlens_now = cu_window_seqlens
            max_seqlens_now = win_max_seqlen

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            # --- Patch.3 ---
            max_seqlen=max_seqlens_now,
            # --- Patch.3 ---
            position_embeddings=position_embeddings,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        merged_sp_padding = sp_padding_size // self.spatial_merge_unit
        merged_hidden_states = gather_seq_scatter_heads(
            merged_hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        )
        if merged_sp_padding > 0:
            merged_hidden_states = unpad_tensor(merged_hidden_states, dim=0, padding_size=merged_sp_padding)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]
        if merged_sp_padding > 0:
            merged_hidden_states = pad_tensor(merged_hidden_states, dim=0, padding_size=merged_sp_padding)
        merged_hidden_states = gather_heads_scatter_seq(
            merged_hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        )
    else:
        merged_hidden_states = merged_hidden_states[reverse_indices, :]
    # --- Patch.1 ---

    # --- Patch.4 ---
    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )
    # --- Patch.4 ---


# ================================================================
# Patch: Qwen2_5_VisionTransformerPretrainedModel.dummy_forward (new)
# 1. add dummy_forward so ranks without pixel_values can still run the
#    visual encoder under FSDP — otherwise reduce-scatter hangs when
#    some ranks get None pixel_values while others have real images.
# 2. SP-aware shape: grid_thw width is scaled by sp_size so the cached
#    dummy batch stays a clean multiple after sequence slicing.
# ================================================================
@config.override_method(
    "Qwen2_5_VisionTransformerPretrainedModel.dummy_forward",
    description="Provide dummy vision forward for FSDP path with SP-aware shape",
)
def qwen2_5_vit_dummy_forward_patched(self):
    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        if getattr(self, "_sp_dummy_data", None) is None:
            # --- Patch.2 ---
            sp_size = get_parallel_state().sp_size
            pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 2 * sp_size, 2]], dtype=torch.int32, device=self.device)
            # --- Patch.2 ---
            self._sp_dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        outputs = self(**self._sp_dummy_data)
    else:
        if getattr(self, "_dummy_data", None) is None:
            pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
            self._dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        outputs = self(**self._dummy_data)
    return outputs
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen2_5_VLModel.get_placeholder_mask
# 1. return raw (image_mask, video_mask) bool tensors without the
#    HF v5 `inputs_embeds` / `*_features` shape-validation branch —
#    VeOmni needs the masks on the *full* all-gathered input_ids,
#    which have a different seq-len than the SP-sliced inputs_embeds.
# Signature keeps the v5 kwargs so any HF-internal caller still works.
# ================================================================
@config.override_method(
    "Qwen2_5_VLModel.get_placeholder_mask",
    description="Return raw image/video placeholder bool masks for VeOmni SP-aware masked_scatter",
)
def qwen2_5_vl_model_get_placeholder_mask_patched(
    self,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor | None = None,
    image_features: torch.FloatTensor | None = None,
    video_features: torch.FloatTensor | None = None,
):
    # --- Patch.1 ---
    special_image_mask = input_ids == self.config.image_token_id
    special_video_mask = input_ids == self.config.video_token_id
    # --- Patch.1 ---
    return special_image_mask, special_video_mask


# ================================================================
# Patch: Qwen2_5_VLModel.forward
# 1. Ulysses SP scatter/gather around the visual-embed masked_scatter
#    (gather_seq_scatter_heads on inputs_embeds + image/video embeds,
#    gather_heads_scatter_seq after fill-back)
# 2. precomputed image/video masks: use kwargs["image_mask"/"video_mask"]
#    when provided by the VeOmni data pipeline; otherwise all-gather
#    input_ids across SP group and recompute masks on the full sequence
# 3. pop ViT-incompatible flash-attn kwargs (cu_seq_lens_q/k,
#    max_length_q/k) before calling ViT/visual; restore onto the
#    language-model kwargs afterwards
# 4. FSDP dummy_forward branch when pixel_values / pixel_values_videos
#    are None on this rank — keeps visual params on the FSDP all-reduce
#    graph via fake_embeds * 0
# 5. honor precomputed 3D position_ids: (bs, 3, L) -> (3, bs, L)
# 6. v5 visual return contract: get_image_features / get_video_features
#    now return BaseModelOutputWithPooling whose pooler_output is a
#    tuple[per-image tensor] — concat into a single tensor
# ================================================================
@config.override_method(
    "Qwen2_5_VLModel.forward",
    description="VeOmni SP + precomputed position-id + dummy-forward multimodal patches",
)
def qwen2_5_vl_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    rope_deltas: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen2_5_VLModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.2 ---
    # Precomputed image/video masks when provided by VeOmni data pipeline,
    # otherwise all-gather input_ids across SP group to compute them.
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    if video_mask is None and image_mask is None:
        input_ids_list = [torch.zeros_like(input_ids) for _ in range(get_parallel_state().sp_size)]
        dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
        image_mask, video_mask = self.get_placeholder_mask(torch.cat(input_ids_list, dim=0))
    # --- Patch.2 ---

    # --- Patch.3 ---
    # Pop ViT-incompatible flash-attn kwargs (they belong to the language model only)
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

    if pixel_values is not None:
        # --- Patch.6 ---
        # v5 get_image_features returns BaseModelOutputWithPooling whose
        # pooler_output is tuple[per-image tensor] after torch.split
        image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
        image_embeds = torch.cat(image_embeds, dim=0)
        # --- Patch.6 ---
        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        # --- Patch.1 ---
        n_image_tokens = image_mask.sum().long().item()
        image_embeds = image_embeds[:n_image_tokens]
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.4 ---

    if pixel_values_videos is not None:
        # --- Patch.6 ---
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
        video_embeds = torch.cat(video_embeds, dim=0)
        # --- Patch.6 ---
        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        # --- Patch.1 ---
        n_video_tokens = video_mask.sum().long().item()
        video_embeds = video_embeds[:n_video_tokens]
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.4 ---

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    else:
        # --- Patch.5 ---
        if position_ids.dim() == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()  # bs, 3, l -> 3, bs, l
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
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


@config.add_helper
def get_position_id(main_func, self, **kwargs):
    # Module-level function so `partial(...)` is picklable for multiprocessing.
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, bs, l)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


# ================================================================
# Patch: Qwen2_5_VLForConditionalGeneration.get_position_id_func (new)
# 1. wrap Qwen2_5_VLModel.get_rope_index so VeOmni's data pipeline can
#    precompute multimodal position_ids in process_sample (on CPU worker
#    processes) — get_position_id is defined as a module-level function
#    so `partial(...)` is picklable for multiprocessing
# 2. overwrite token ids with VeOmni constants (IMAGE_INPUT_INDEX /
#    VIDEO_INPUT_INDEX) so input_ids produced by our data pipeline match
# ================================================================
@config.override_method(
    "Qwen2_5_VLForConditionalGeneration.get_position_id_func",
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)
def qwen2_5_vl_get_position_id_func_patched(self):
    # --- Patch.1 ---
    fake_config = copy.copy(self.config)
    # --- Patch.2 ---
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    # --- Patch.2 ---
    fake_model = SimpleNamespace(config=fake_config)
    return partial(get_position_id, Qwen2_5_VLModel.get_rope_index, fake_model)  # noqa: F821
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen2_5_VLForConditionalGeneration.forward
# 1. use the unified VeOmni fused loss_function (handles Ulysses
#    internally, takes hidden_states + lm_head weights instead of
#    pre-computed logits) — avoids materializing full-vocab logits
#    when labels are provided
# 2. drop the HF v5 logits-first path — only compute logits when
#    labels is None (inference); otherwise let loss_function fuse
#    matmul + cross-entropy
# ================================================================
@config.override_method(
    "Qwen2_5_VLForConditionalGeneration.forward",
    description="Use VeOmni unified fused loss_function path",
)
def qwen2_5_vl_for_conditional_generation_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    rope_deltas: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    outputs: Qwen2_5_VLModelOutputWithPast = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    log_probs = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, log_probs = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, log_probs = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    output = Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
    output.log_probs = log_probs
    return output
