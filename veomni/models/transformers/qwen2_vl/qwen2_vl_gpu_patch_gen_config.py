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
Patch configuration for Qwen2-VL transformers>=5.0.0 code generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen2_vl.qwen2_vl_gpu_patch_gen_config -o veomni/models/transformers/qwen2_vl/generated --diff
"""

import copy
from functools import partial
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    is_flash_attention_requested,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
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
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX


config = PatchConfig(
    source_module="transformers.models.qwen2_vl.modeling_qwen2_vl",
    target_file="patched_modeling_qwen2_vl_gpu.py",
    description="Qwen2-VL with VeOmni v5 compatibility and LigerKernel GPU replacements",
)

config.add_import("copy", is_from_import=False)
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_heads_scatter_seq", "gather_seq_scatter_heads", "pad_tensor", "sp_pad_and_slice"],
)
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])


@config.override_method(
    "VisionAttention.forward",
    description="Use VeOmni varlen attention types and precomputed max_seqlen to avoid per-layer cpu-gpu sync",
)
def vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
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
        # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
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


@config.override_method(
    "Qwen2VisionTransformerPretrainedModel.forward",
    description="VeOmni SP + Precompute max_seqlen and apply SP position-embedding slicing in vision forward",
)
def qwen2_vit_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> BaseModelOutputWithPooling:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # VeOmni SP Patch Start:
    if get_parallel_state().sp_enabled:
        unpadded_dim_size = cu_seqlens[-1]
        sp_padding_size = hidden_states.shape[0] * get_parallel_state().sp_size - unpadded_dim_size
        emb = pad_tensor(emb, dim=0, padding_size=sp_padding_size)
        emb = sp_pad_and_slice(emb, dim=0)

        if sp_padding_size > 0:
            new_cumsum = cu_seqlens[-1] + sp_padding_size
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
    # VeOmni SP Patch End.

    position_embeddings = (emb.cos(), emb.sin())
    # use precomputed max_seqlen in advance to avoid per-layer cpu-gpu sync
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()

    for blk in self.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)

    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )


@config.override_method(
    "Qwen2VisionTransformerPretrainedModel.dummy_forward",
    description="Provide dummy vision forward for FSDP path with SP-aware shape",
)
def qwen2_vit_dummy_forward_patched(self):
    if get_parallel_state().sp_enabled:
        if getattr(self, "_sp_dummy_data", None) is None:
            sp_size = get_parallel_state().sp_size
            pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 2 * sp_size, 2]], dtype=torch.int32, device=self.device)
            self._sp_dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        outputs = self(**self._sp_dummy_data)
    else:
        if getattr(self, "_dummy_data", None) is None:
            pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
            self._dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        outputs = self(**self._dummy_data)

    return outputs


@config.override_method(
    "Qwen2VLModel.get_video_features",
    description="Simplify visual feature extraction and handle BaseModelOutputWithPooling",
)
def qwen2vl_model_get_video_features_patched(
    self,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | BaseModelOutputWithPooling:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
        The tensors corresponding to the input videos.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    vision_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw, return_dict=True, **kwargs)

    return vision_outputs


@config.override_method(
    "Qwen2VLModel.get_image_features",
    description="Simplify visual feature extraction and handle BaseModelOutputWithPooling",
)
def qwen2vl_model_get_image_features_patched(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | BaseModelOutputWithPooling:
    r"""
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
        The tensors corresponding to the input images.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)

    return vision_outputs


@config.override_method(
    "Qwen2VLModel.forward",
    description="Apply VeOmni SP + precomputed position-id + dummy-forward multimodal patches",
)
def qwen2vl_model_forward_patched(
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
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen2VLModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Extract image and video masks from kwargs
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # VeOmni SP Patch
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output

        # VeOmni SP
        if get_parallel_state().sp_enabled:
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )

        image_embeds = image_embeds[: image_mask.sum()]
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    elif get_parallel_state().fsdp_enabled:
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output

        # VeOmni SP
        if get_parallel_state().sp_enabled:
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )

        video_embeds = video_embeds[: video_mask.sum()]
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    elif get_parallel_state().fsdp_enabled:
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds

    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    # Use VeOmni precomputed position ids
    else:
        if position_ids.shape[1] == 3:  # bs, 3, l
            position_ids = position_ids.transpose(0, 1).contiguous()  # bs, 3, l -> 3, bs, l

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

    output = Qwen2VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


# ================================================================
# Patch: Qwen2VLForConditionalGeneration.get_position_id_func
# ================================================================
@config.override_method(
    "Qwen2VLForConditionalGeneration.get_position_id_func",
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)
def qwen2vl_get_position_id_func_patched(self):
    def get_position_id(main_func, self, **kwargs):
        position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, 1, l), rope_deltas (1, 1)
        return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}

    fake_config = copy.copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = SimpleNamespace(config=fake_config)
    return partial(get_position_id, Qwen2VLModel.get_rope_index, fake_model)


@config.override_method(
    "Qwen2VLForConditionalGeneration.forward",
    description="Use VeOmni unified loss path for Qwen2VLForConditionalGeneration.forward",
)
def qwen2vl_for_conditional_generation_forward_patched(
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
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen2VLCausalLMOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # Extract image and video masks from kwargs
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)

    outputs: Qwen2VLModelOutputWithPast = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_mask=image_mask,
        video_mask=video_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
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
    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
