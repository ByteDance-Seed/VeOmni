# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as hf_qwen25vl
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLVisionAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as _Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _Qwen2_5_VLModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
)

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    pad_tensor,
    sp_pad_and_slice,
    unpad_tensor,
)
from ....utils import logging
from ....utils.device import IS_NPU_AVAILABLE


logger = logging.get_logger(__name__)


# ================================================================
# Patch: Qwen2_5_VLVisionAttention.forward
# 1. use precomputed max_seqlen in advance to avoid per-layer cpu-gpu sync
# ================================================================
def Qwen2_5_VLVisionAttention_forward(
    self: Qwen2_5_VLVisionAttention,
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

    if self.config._attn_implementation == "flash_attention_2":
        # Flash Attention 2: Use cu_seqlens for variable length attention
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
# Patch: Qwen2_5_VisionTransformerPretrainedModel.forward
# 1. use all-to-all to get full sequence of hidden_states for window attention
# 2. sp patch cu_seqlens & position_embeds
# 3. calculate max_seqlen from cu_seqlens here to avoid per layer CPU-GPU sync
# 4. move cu_seqlens to cpu when using NPU to avoid per layer CPU-GPU sync when using FA
# ================================================================
def Qwen2_5_VisionTransformerPretrainedModel_forward(
    self: Qwen2_5_VisionTransformerPretrainedModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
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
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # --- Patch.1 ---
    unpadded_dim_size = cu_seqlens[-1]
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

    # --- Patch.4 ---
    if IS_NPU_AVAILABLE:
        cu_seqlens = cu_seqlens.cpu()
    # --- Patch.4 ---

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
            max_seqlen=max_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        sp_padding_size = hidden_states.size(0) - unpadded_dim_size
        hidden_states = gather_seq_scatter_heads(
            hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        )
        if sp_padding_size > 0:
            hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
    # --- Patch.1 ---

    hidden_states = hidden_states[reverse_indices, :]

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        if sp_padding_size > 0:
            hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
        hidden_states = gather_heads_scatter_seq(
            hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    return hidden_states


# ================================================================
# Patch: Qwen2_5_VisionTransformerPretrainedModel.dummy_forward
# 1. add dummy_forward to avoid FSDP reduce-scatter hang when some ranks
# get None pixel_values while others get valid pixel_values
# ================================================================
# --- Patch.1 ---
def Qwen2_5_VisionTransformerPretrainedModel_dummy_forward(self: Qwen2_5_VisionTransformerPretrainedModel):
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


# --- Patch.1 ---


# ================================================================
# Patch: Qwen2_5_VLModel
# 1. skip torch.split in get_image_features
# 2. patch get_placeholder_mask for veomni usage
# 3. sequence parallel forward for sp sliced input_embeds & image_mask
# & video_mask $ deepstack embeds
# 4. dummy forward patch
# 5. handle precomputed position_ids with shape (bs, dim, seq_len)
# 6. Use precomputed flash attention kwargs to avoid CPU-GPU sync
# ================================================================
class Qwen2_5_VLModel(_Qwen2_5_VLModel):
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        # --- Patch.1 ---
        # split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        # video_embeds = torch.split(video_embeds, split_sizes)
        # --- Patch.1 ---
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # --- Patch.1 ---
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        # --- Patch.1 ---
        return image_embeds

    def get_placeholder_mask(self, input_ids: torch.LongTensor, **kwargs):
        # --- Patch.2 ---
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        # --- Patch.2 ---
        return special_image_mask, special_video_mask

    @auto_docstring
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
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # --- Patch.3 ---
        # we use the pre-computed image and video mask to support ulysses
        image_mask = kwargs.get("image_mask", None)
        video_mask = kwargs.get("video_mask", None)

        # if None, all gather sp group input_ids and calculate mask
        if video_mask is None and image_mask is None:
            input_ids_list = [torch.zeros_like(input_ids) for i in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            image_mask, video_mask = self.get_placeholder_mask(torch.cat(input_ids_list, dim=0))
        # --- Patch.3 ---

        # --- Patch.6 ---
        # Pop flash attention kwargs for ViT, they should only be used for language model
        # because Qwen3L ViT input images seq lens should be computed during ViT forward using grid_thw
        flash_attn_kwargs = {}
        for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
            if key in kwargs:
                flash_attn_kwargs[key] = kwargs.pop(key)
        # --- Patch.6 ---

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )
        # --- Patch.3 ---

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            # --- Patch.3 ---
            if get_parallel_state().sp_enabled:
                image_embeds = gather_seq_scatter_heads(
                    image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )
            # --- Patch.3 ---
            n_image_tokens = image_mask.sum().long().item()
            image_embeds = image_embeds[:n_image_tokens]
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        elif get_parallel_state().fsdp_enabled:
            # --- Patch.4 ---
            fake_embeds = self.visual.dummy_forward().mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.4 ---
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # --- Patch.3 ---
            if get_parallel_state().sp_enabled:
                video_embeds = gather_seq_scatter_heads(
                    video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )
            # --- Patch.3 ---
            n_video_tokens = video_mask.sum().long().item()
            video_embeds = video_embeds[:n_video_tokens]
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        elif get_parallel_state().fsdp_enabled:
            # --- Patch.4 ---
            fake_embeds = self.visual.dummy_forward().mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.4 ---

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            )
        # --- Patch.3 ---

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)
        else:
            # --- Patch.5 ---
            if position_ids.dim() == 3 and position_ids.shape[1] == 3:
                position_ids = position_ids.transpose(0, 1).contiguous()  # bs, dim, l -> dim, bs, l
            # --- Patch.5 ---

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            position_ids = sp_pad_and_slice(position_ids, dim=-1)
        # --- Patch.3 ---

        # --- Patch.6 ---
        kwargs.update(flash_attn_kwargs)
        # --- Patch.6 ---

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


# ================================================================
# Patch: Qwen2_5_VLForConditionalGeneration
# 1. wrapped Qwen2_5_VLModel.get_rope_index to use in process_sample for obtaining position_ids in advance
# 2. use the unified loss function to handle Ulysses internally to reduce redudnecy code
# ================================================================


# --- Patch.1 ---
def get_position_id(main_func, self, **kwargs):
    # must be a global func for multiproceesing serialize
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, bs, l)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


# --- Patch.1 ---


class Qwen2_5_VLForConditionalGeneration(_Qwen2_5_VLForConditionalGeneration):
    # --- Patch.1 ---
    def get_position_id_func(self):
        fake_model = SimpleNamespace(config=self.config)
        return partial(get_position_id, Qwen2_5_VLModel.get_rope_index, fake_model)

    # --- Patch.1 ---

    @can_return_tuple
    @auto_docstring
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
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
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

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        # --- Patch.2 ---
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
            logits = self.lm_head(hidden_states)
        # --- Patch.2 ---

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


def apply_veomni_qwen25_vl_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen2.5_VL.")
    hf_qwen25vl.Qwen2_5_VLVisionAttention.forward = Qwen2_5_VLVisionAttention_forward
    hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_forward
    hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel.dummy_forward = (
        Qwen2_5_VisionTransformerPretrainedModel_dummy_forward
    )
    hf_qwen25vl.Qwen2_5_VLModel = Qwen2_5_VLModel
    hf_qwen25vl.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
