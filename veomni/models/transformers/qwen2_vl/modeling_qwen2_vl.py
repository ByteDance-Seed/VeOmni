# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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

"""PyTorch Qwen2-VL model."""

import copy
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_vl import modeling_qwen2_vl as hf_qwen2vl
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2VLAttention,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLModelOutputWithPast,
    Qwen2VLTextConfig,
    TransformersKwargs,
    VisionAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel as _Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer as _Qwen2VLDecoderLayer
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration as _Qwen2VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as _Qwen2VLModel
from transformers.processing_utils import Unpack

from ....data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    pad_tensor,
    sp_pad_and_slice,
)
from ....utils import logging
from ....utils.device import IS_CUDA_AVAILABLE
from ..attention_utils import VARLEN_ATTENTION_TYPES


logger = logging.get_logger(__name__)


# ================================================================
# Patch: VisionAttention.forward
# 1. veomni varlen attention types
# 2. use precomputed max_seqlen in advance to avoid per-layer cpu-gpu sync
# ================================================================
def VisionAttention_forward(
    self: VisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # --- Patch.1 ---
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
        # --- Patch.1 ---
        # Flash Attention 2: Use cu_seqlens for variable length attention
        # --- Patch.2 ---
        # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        # --- Patch.2 ---
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
        # Other implementations: Process each chunk separately
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
# Patch: Qwen2VisionTransformerPretrainedModel
# 1. use precomputed max_seqlen in advance to avoid per-layer cpu-gpu sync
# 2. sp slice position embeddings
# 3. dummy forward
# ================================================================
class Qwen2VisionTransformerPretrainedModel(_Qwen2VisionTransformerPretrainedModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
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
        # --- Patch.2 ---
        if get_parallel_state().sp_enabled:
            unpadded_dim_size = cu_seqlens[-1]
            sp_padding_size = hidden_states.shape[0] * get_parallel_state().sp_size - unpadded_dim_size
            emb = pad_tensor(emb, dim=0, padding_size=sp_padding_size)
            emb = sp_pad_and_slice(emb, dim=0)

            if sp_padding_size > 0:
                new_cumsum = cu_seqlens[-1] + sp_padding_size
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        # --- Patch.2 ---
        position_embeddings = (emb.cos(), emb.sin())
        # --- Patch.1 ---
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
        # --- Patch.1 ---
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return self.merger(hidden_states)

    # --- Patch.3 ---
    def dummy_forward(self):
        if get_parallel_state().sp_enabled:
            if getattr(self, "_sp_dummy_data", None) is None:
                sp_size = get_parallel_state().sp_size
                pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
                grid_thw = torch.tensor([[1, 2 * sp_size, 2]], dtype=torch.int32, device=self.device)
                self._sp_dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
            return self(**self._sp_dummy_data)
        else:
            if getattr(self, "_dummy_data", None) is None:
                pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
                grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
                self._dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
            return self(**self._dummy_data)

    # --- Patch.3 ---


# ================================================================
# Patch: Qwen2VLDecoderLayer
# 1. veomni varlen attention types
# ================================================================
class Qwen2VLDecoderLayer(_Qwen2VLDecoderLayer):
    def __init__(self, config: Qwen2VLTextConfig, layer_idx: int):
        super(_Qwen2VLDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        # --- Patch.1 ---
        if config.use_sliding_window and config._attn_implementation not in VARLEN_ATTENTION_TYPES:
            # --- Patch.1 ---
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2VLAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]


# ================================================================
# Patch: Qwen2VLModel
# 1. sp patch
# 2. dummy forward
# 3. use veomni precompute position ids
# 4. simplify get visual embeds
# ================================================================
class Qwen2VLModel(_Qwen2VLModel):
    # --- Patch.4 ---
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds

    # --- Patch.4 ---

    # --- Patch.4 ---
    def get_image_features(
        self, pixel_values_images: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None
    ):
        pixel_values_images = pixel_values_images.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values_images, grid_thw=image_grid_thw)
        return image_embeds

    # --- Patch.4 ---

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_mask: Optional[torch.FloatTensor] = None,
        video_mask: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # --- Patch.1 ---
            if get_parallel_state().sp_enabled:
                inputs_embeds = gather_seq_scatter_heads(
                    inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
                )
            # --- Patch.1 ---

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                # --- Patch.1 ---
                if get_parallel_state().sp_enabled:
                    image_embeds = gather_seq_scatter_heads(
                        image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                    )
                # --- Patch.1 ---
                image_embeds = image_embeds[: image_mask.sum()]
                image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            # --- Patch.2 ---
            elif get_parallel_state().fsdp_enabled:
                fake_embeds = self.visual.dummy_forward().mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.2 ---

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                # --- Patch.1 ---
                if get_parallel_state().sp_enabled:
                    video_embeds = gather_seq_scatter_heads(
                        video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                    )
                # --- Patch.1 ---
                video_embeds = video_embeds[: video_mask.sum()]
                video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            # --- Patch.2 ---
            elif get_parallel_state().fsdp_enabled:
                fake_embeds = self.visual.dummy_forward().mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.2 ---

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            )
        # --- Patch.1 ---

        if position_ids is None:
            if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids + delta.to(position_ids.device)
        # --- Patch.3 ---
        else:
            if position_ids.shape[1] == 3:  # bs, 3, l
                position_ids = position_ids.transpose(0, 1).contiguous()  # bs, 3, l -> 3, bs, l
        # --- Patch.3 ---

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            position_ids = sp_pad_and_slice(position_ids, dim=-1)
        # --- Patch.1 ---

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
# Patch: Qwen2VLForConditionalGeneration
# 1. precompute position ids
# 2. use the unified loss function
# 3. veomni unified multimodal tokens
# ================================================================
# --- Patch.1 ---
def get_position_id(main_func, self, **kwargs):
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, 1, l), rope_deltas (1, 1)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}


# --- Patch.1 ---


class Qwen2VLForConditionalGeneration(_Qwen2VLForConditionalGeneration):
    # --- Patch.1 ---
    def get_position_id_func(self):
        fake_config = copy.copy(self.config)
        # --- Patch.3 ---
        fake_config.image_token_id = IMAGE_INPUT_INDEX
        fake_config.video_token_id = VIDEO_INPUT_INDEX
        # --- Patch.3 ---
        fake_model = SimpleNamespace(config=fake_config)
        return partial(get_position_id, Qwen2VLModel.get_rope_index, fake_model)

    # --- Patch.1 ---

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
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

        hidden_states = outputs[0]

        # --- Patch.2 ---
        loss = None
        logits = None
        if labels is not None:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
        # --- Patch.2 ---

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


def apply_veomni_qwen2vl_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen2-VL.")
    hf_qwen2vl.Qwen2VLForConditionalGeneration = Qwen2VLForConditionalGeneration
    hf_qwen2vl.Qwen2VLModel = Qwen2VLModel
    hf_qwen2vl.Qwen2VisionTransformerPretrainedModel = Qwen2VisionTransformerPretrainedModel
    hf_qwen2vl.Qwen2VLDecoderLayer = Qwen2VLDecoderLayer
    hf_qwen2vl.VisionAttention.forward = VisionAttention_forward

    if IS_CUDA_AVAILABLE:
        from .gpu_patch import apply_veomni_qwen2vl_gpu_patch

        apply_veomni_qwen2vl_gpu_patch()
