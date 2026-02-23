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
#
# Patch for transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe
# Adds support for: Sequence Parallelism (SP), FSDP, Expert Parallelism (EP),
# Liger kernel, fused MoE, pre-computed masks,
# VeOmni data constants, and multiprocessing-compatible position ID generation.

from functools import partial
from types import SimpleNamespace
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as hf_qwen3_omni_moe
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.processing_utils import Unpack

from ....data.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    reduce_sequence_parallel_loss,
    slice_input_tensor,
    slice_position_embedding,
    sp_pad_and_slice,
    unpad_tensor,
)
from ....distributed.sequence_parallel.ulysses import _Gather
from ....ops import fused_moe_forward
from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available


if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore

logger = logging.get_logger(__name__)


# ================================================================
# PATCH: Qwen3OmniMoeVisionEncoder.forward
# 1. Support SP for position embeddings (slice pos_embeds with sp_pad_and_slice)
# 2. Support SP for rotary position embeddings
# 3. Pad cu_seqlens when using SP to match padded hidden_states
# ================================================================
def qwen3_omni_moe_vision_encoder_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder,
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

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    # --- Patch.1 ---
    # Modification: slice pos embedding if using sp to let sharded hidden_states get its corresponding pos embedding
    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    if sp_group is not None:
        # We need to do padding here because of hidden_states did padding with pad_scale=4
        pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
    # --- Patch.1 ---
    hidden_states = hidden_states + pos_embeds

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    # --- Patch.2 ---
    # Modification: Get before-sliced full seq from cu_seqlens
    # total_seq_len should equal to seq_len when not using SP
    total_seq_len = cu_seqlens[-1]
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # Modification: slice pos embedding when using sp to let sp-sliced hidden_states get its corresponding pos embedding
    if sp_group is not None:
        cos, sin = position_embeddings
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
        position_embeddings = (cos, sin)

    # Modification: pad cu_seqlens when using SP to match the padded hidden_states
    if sp_group is not None:
        ps = get_parallel_state()
        sp_size = getattr(ps, "sp_size", 1)
        # Calculate the last one padding seq_len : seq_len*sp_size - total_seq_len
        pad_seq_len = seq_len * sp_size - total_seq_len.item()
        if pad_seq_len > 0:
            # Add this extra sequence to cu_seqlens with the padding length
            new_cumsum = cu_seqlens[-1] + pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
    # --- Patch.2 ---

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)

    return hidden_states, deepstack_feature_lists


# ================================================================
# PATCH: Qwen3OmniMoeVisionEncoder.dummy_forward (NEW)
# Prevent FSDP reduce-scatter hang when some ranks get None pixel_values
# while others get valid pixel_values
# ================================================================
def qwen3_omni_moe_vision_encoder_dummy_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder,
):
    if get_parallel_state().sp_enabled:
        sp_size = get_parallel_state().sp_size
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        # If using SP, pixel_values is sliced but grid_thw is not
        grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
    else:
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
    return self(**dummy_data)


# ================================================================
# PATCH: Qwen3OmniMoeAudioEncoder.forward
# 1. SP support: gather input_features along time dim, strip SP padding before chunking
# 2. SP support: slice hidden_states before encoder layers, extend cu_seqlens for padded tail
# ================================================================
def qwen3_omni_moe_audio_encoder_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder,
    input_features,
    feature_lens=None,
    aftercnn_lens=None,
):
    aftercnn_lens = hf_qwen3_omni_moe._get_feat_extract_output_lengths(feature_lens)
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = self.n_window * 2

    # --- Patch.1 ---
    # Modification: SP support — input_features is (num_mel_bins, total_len);
    # gather along the time dimension (dim=1) and strip any SP padding before chunking.
    if get_parallel_state().sp_enabled:
        unpadded_input_len = torch.sum(chunk_lengths)
        input_features = gather_outputs(input_features, gather_dim=1, group=get_parallel_state().sp_group)
        sp_input_padding = input_features.size(1) - unpadded_input_len
        if sp_input_padding > 0:
            input_features = unpad_tensor(input_features, dim=1, padding_size=sp_input_padding)
    # --- Patch.1 ---

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = hf_qwen3_omni_moe._get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)
    # Split to chunk to avoid OOM during convolution
    padded_embeds = []
    for chunk in padded_feature.split(self.conv_chunksize, dim=0):
        padded_embed = F.gelu(self.conv2d1(chunk))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
        padded_embeds.append(padded_embed)
    padded_embed = torch.cat(padded_embeds, dim=0)
    b, c, f, t = padded_embed.size()
    padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    positional_embedding = (
        self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

    # --- Patch.2 ---
    # Modification: SP support — slice hidden_states along seq dim (dim=0) before encoder layers.
    # Track total unpadded length to compute SP padding size for cu_seqlens extension.
    if get_parallel_state().sp_enabled:
        unpadded_hidden_len = cu_seqlens[-1]
        hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
        # Extend cu_seqlens to cover the padded tail on the last rank if needed
        pad_seq_len = hidden_states.size(0) * get_parallel_state().sp_size - unpadded_hidden_len
        if pad_seq_len > 0:
            cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)], dim=0)
    # --- Patch.2 ---

    for encoder_layer in self.layers:
        layer_outputs = encoder_layer(
            hidden_states,
            cu_seqlens,
        )
        hidden_states = layer_outputs[0]

    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return BaseModelOutput(last_hidden_state=hidden_states)


# ================================================================
# PATCH: Qwen3OmniMoeAudioEncoder.dummy_forward (NEW)
# Prevent FSDP reduce-scatter hang when some ranks have no audio data
# while others get valid audio data
# ================================================================
def qwen3_omni_moe_audio_encoder_dummy_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder,
):
    """
    Dummy forward to avoid FSDP reduce-scatter hang when some ranks have no audio data.
    input_features shape is (num_mel_bins, total_len), feature_lens is (num_audios,).
    """
    if getattr(self, "_dummy_data", None) is None:
        # Minimal valid input: one audio clip of length n_window*2 (smallest non-zero chunk)
        min_len = self.n_window * 2
        input_features = torch.zeros((self.num_mel_bins, min_len), dtype=self.dtype, device=self.device)
        feature_lens = torch.tensor([min_len], dtype=torch.long, device=self.device)
        self._dummy_data = {
            "input_features": input_features,
            "feature_lens": feature_lens,
        }
    return self(**self._dummy_data)


# ================================================================
# PATCH: Qwen3OmniMoeThinkerTextModel.forward
# 1. Support SP for position embeddings via slice_position_embedding
# ================================================================
def qwen3_omni_moe_thinker_text_model_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    r"""
    visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
        The mask of the visual positions.
    deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
        The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
        The feature is extracted from the different visual encoder layers, and fed to the decoder
        hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # --- Patch.1 ---
    # Modification: slice pos embedding when using sp to let sp-sliced hidden_states get its corresponding pos_embedding
    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    if sp_group is not None:
        position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
    # --- Patch.1 ---

    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ================================================================
# PATCH: Qwen3OmniMoeThinkerTextModel._deepstack_process
# 1. Handle None visual_pos_masks for FSDP hang prevention
#    (add 0.0 to hidden_states to keep FSDP backward reduce scatter working)
# 2. Mask format changed: visual_pos_masks is now pre-computed without extra dim
# ================================================================
def qwen3_omni_moe_thinker_text_model_deepstack_process(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel,
    hidden_states,
    visual_pos_masks,
    visual_embeds,
):
    # --- Patch.1 ---
    # Modification: Handle case when visual_pos_masks is None (both image and video pixel_values are None)
    # Still call this operation but just add 0.0 to hidden_states to avoid FSDP reduce scatter stuck
    if visual_pos_masks is None:
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + visual_embeds.mean() * 0.0
        return hidden_states
    # --- Patch.1 ---

    # --- Patch.2 ---
    # Modification: removed visual_pos_masks[..., 0] since mask is now pre-computed in correct format
    # Also handle 3D masks (bsz, seq_len, 1) by squeezing the trailing dim
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    if visual_pos_masks.ndim == 3:
        visual_pos_masks = visual_pos_masks[..., 0]
    # --- Patch.2 ---
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


# ================================================================
# PATCH: Qwen3OmniMoeThinkerForConditionalGeneration.__init__
# 1. Use VeOmni data constants (IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX, AUDIO_INPUT_INDEX)
#    for token indices instead of config token_id values
# ================================================================
def qwen3_omni_moe_thinker_init(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration,
    config,
):
    super(hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration, self).__init__(config)
    self.audio_tower = hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder._from_config(config.audio_config)
    self.visual = hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder._from_config(config.vision_config)
    self.vocab_size = config.text_config.vocab_size
    self.model = hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel._from_config(config.text_config)
    from torch import nn

    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    self.spatial_merge_size = config.vision_config.spatial_merge_size
    self.rope_deltas = None
    self.num_experts = config.text_config.num_experts
    self.num_experts_per_tok = config.text_config.num_experts_per_tok

    # --- Patch.1 ---
    # Modification: Use VeOmni data constants for multimodal token indices
    self.image_token_index = IMAGE_INPUT_INDEX
    self.video_token_index = VIDEO_INPUT_INDEX
    self.audio_token_index = AUDIO_INPUT_INDEX
    # --- Patch.1 ---
    self.post_init()


# ================================================================
# NEW: get_position_id
# Global function for multiprocessing serialization to generate position_ids
# ================================================================
def get_position_id(main_func, self, **kwargs):
    """
    This function is used during the data preprocessing stage to generate position_ids
    and associated parameters (e.g., rope_deltas) for a **single sample** (bs = 1).
    This function is a global function for multiprocessing serialization.
    Args:
        main_func: model.get_position_id
        self: An object holding model-specific information (e.g., SimpleNamespace(config=...)).
        **kwargs: Additional arguments passed to `main_func` (e.g., input_ids).
    Returns:
        dict:
            - "position_ids": Tensor of shape (dim, l), with the batch dimension squeezed.
            - other necessary parameters with the batch dimension squeezed (e.g., rope_deltas).

    Example usage:
        class Model:
            def get_position_id_func(self):  # Used in data_transform during training
                fake_model = SimpleNamespace(config=self.config)
                return partial(get_position_id, main_func, fake_model)

        model = Model()
        func = model.get_position_id_func()
        position_func_returns = func(input_ids=input_ids.unsqueeze(0), **kwargs)
        position_ids = position_func_returns['position_ids']  # shape: (dim, l)

    If a model does not implement `get_position_id_func()`, a default fallback for position_ids can be:
        position_id_returns = {
            "position_ids": torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)  # shape: (dim, l)
        }
    """
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, 1, l), rope_deltas (1, 1)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}


# ================================================================
# PATCH: Qwen3OmniMoeThinkerForConditionalGeneration.get_position_id_func (NEW)
# Provides a serializable function for multiprocessing data preprocessing
# ================================================================
def qwen3_omni_moe_thinker_get_position_id_func(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration,
):
    fake_model = SimpleNamespace(
        config=self.config,
        image_token_index=self.image_token_index,
        video_token_index=self.video_token_index,
        audio_token_index=self.audio_token_index,
        spatial_merge_size=self.spatial_merge_size,
        get_llm_pos_ids_for_vision=partial(
            hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision, None
        ),
        get_chunked_index=partial(
            hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_chunked_index, None
        ),
    )
    return partial(
        get_position_id,
        hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index,
        fake_model,
    )


# ================================================================
# PATCH: Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features
# 1. Remove NotImplementedError for SP (audio SP is now handled inside audio_tower)
# 2. Support feature_attention_mask unpacking into flat (num_mel_bins, total_len) format
# 3. Handle pre-processed 2D input compatibility mode
# ================================================================
def qwen3_omni_moe_thinker_get_audio_features(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration,
    input_features: torch.FloatTensor,
    feature_attention_mask: Optional[torch.LongTensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
):
    """
    Encodes audios into continuous embeddings that can be forwarded to the language model.

    Args:
        input_features (`torch.FloatTensor`):
            The tensors corresponding to the input audios.
        feature_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
    """
    # Modification: unpack feature_attention_mask into flat (num_mel_bins, total_len) format
    # expected by the audio_tower, then delegate. SP is handled inside audio_tower.
    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    else:
        # Compatibility: handle pre-processed (total_len, num_mel_bins) input without mask
        if input_features.ndim == 2 and input_features.size(-1) == self.audio_tower.num_mel_bins:
            input_features = input_features.transpose(0, 1)  # -> (num_mel_bins, total_len)

    if audio_feature_lengths is not None:
        feature_lens = audio_feature_lengths
    else:
        feature_lens = torch.tensor([input_features.size(1)], dtype=torch.long, device=input_features.device)
    audio_outputs = self.audio_tower(
        input_features,
        feature_lens=feature_lens,
    )
    audio_features = audio_outputs.last_hidden_state

    return audio_features


# ================================================================
# PATCH: Qwen3OmniMoeThinkerForConditionalGeneration.forward
# 1. Use pre-computed image_mask/video_mask/audio_mask from kwargs (avoid all-gather for complete mask in SP)
# 2. Pop flash attention kwargs for ViT (ViT computes its own cu_seqlens from grid_thw)
# 3. SP: gather_seq_scatter_heads for input/image/video/audio embeddings
# 4. Dummy ViT/audio forward for FSDP when pixel_values/input_features is None
# 5. SP: gather_heads_scatter_seq after multimodal merging
# 6. SP: all_gather deepstack embeddings + rank-specific slicing
# 7. FSDP: fake_deepstack handling when both pixel_values and pixel_values_videos are None
# 8. Custom loss computation with Liger kernel + SP loss reduction
# 9. Handle pre-computed position_ids from data preprocessing (bs, 3, L) -> (3, bs, L)
# ================================================================
def qwen3_omni_moe_thinker_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration,
    input_ids=None,
    input_features=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    feature_attention_mask=None,
    audio_feature_lengths=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    rope_deltas=None,
    labels=None,
    use_cache=None,
    output_router_logits: Optional[bool] = None,
    use_audio_in_video=None,
    cache_position=None,
    video_second_per_grid=None,
    **kwargs,
) -> Union[tuple, hf_qwen3_omni_moe.Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
        Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
        The length of feature shape of each audio in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    use_audio_in_video (`bool`, *optional*):
        Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
    video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
        Number of seconds per grid for each video, used for temporal feature mapping.
    """
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
    )

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Modification: we use the pre-computed image, video and audio mask to support ulysses
    assert "image_mask" in kwargs, "image_mask should have already been computed in process_sample"
    assert "video_mask" in kwargs, "video_mask should have already been computed in process_sample"
    assert "audio_mask" in kwargs, "audio_mask should have already been computed in process_sample"
    image_mask = kwargs["image_mask"]
    video_mask = kwargs["video_mask"]
    audio_mask = kwargs["audio_mask"]

    # --- Patch.2 ---
    # Modification: Pop flash attention kwargs for ViT, they should only be used for language model
    # Qwen3L ViT input images seq lens should be computed during ViT forward using grid_thw
    # https://github.com/huggingface/transformers/blob/94df0e65602922be2831b3faa457a2bde78b936b/src/transformers/modeling_flash_attention_utils.py#L432-L450
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)
    # --- Patch.2 ---

    # --- Patch.3 ---
    if self.training and get_parallel_state().sp_enabled:
        # (batch_size, seq_len // sp_size, hidden_size) to  (batch_size, seq_len, hidden_size // sp_size)
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )
    # --- Patch.3 ---

    # 2. Merge text , audios , image and video
    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---
        # Modification: SP patch — audio_tower returns sliced-seq features; gather along seq and
        # scatter along hidden dim to match inputs_embeds layout during fill-back.
        if self.training and get_parallel_state().sp_enabled:
            audio_features = gather_seq_scatter_heads(
                audio_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
            )
        # --- Patch.3 ---
        # Modification: drop any padding tokens beyond the actual audio placeholder count
        n_audio_tokens = audio_mask.sum().long().item()
        audio_features = audio_features[:n_audio_tokens]
        audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask_expanded, audio_features)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.5 ---
        # Modification: dummy audio tower forward to keep FSDP reduce-scatter in sync when
        # some ranks have no audio data while others do.
        fake_audio = self.audio_tower.dummy_forward().last_hidden_state.mean() * 0.0
        fake_audio = fake_audio.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_audio
        # --- Patch.5 ---

    # Initialize fake_deepstack to None
    fake_deepstack = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---
        # Modification: sp patch
        if self.training and get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        # --- Patch.3 ---
        # Original: We calculated the special_image_mask in the forward pass,
        # but now we pre-compute it and pass it in as image_mask to avoid
        # all-gather to get complete image_mask info when using sequence parallel
        # image_mask, _, _ = self.get_placeholder_mask(
        #     input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        # )

        # --- Patch.4 ---
        # Modification: Get the num of image tokens from the pre-computed image_mask
        # And reshape the masks to match the shape of inputs_embeds
        n_image_tokens = image_mask.sum().long().item()
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)

        # Modification: Slice tensor to drop any padded image tokens
        image_embeds = image_embeds[:n_image_tokens]
        deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
        # --- Patch.4 ---
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.5 ---
        # Modification: add dummy ViT forward to avoid FSDP reduce-scatter hang
        # when some ranks get None pixel_values while others get valid pixel_values
        fake_embeds, fake_deepstack = self.visual.dummy_forward()
        fake_embeds = fake_embeds.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.5 ---

    if pixel_values_videos is not None:
        video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---
        # Modification: sequence parallel patch for video embeds
        if self.training and get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        # --- Patch.3 ---
        # _, video_mask, _ = self.get_placeholder_mask(
        #     input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        # )

        # --- Patch.4 ---
        # Modification: Get the num of video tokens from the pre-computed video_mask
        # And reshape the masks to match the shape of inputs_embeds
        n_video_tokens = video_mask.sum().long().item()
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)

        # Modification: Slice tensor to drop any padded video tokens
        video_embeds = video_embeds[:n_video_tokens]
        deepstack_video_embeds = video_embeds_multiscale
        deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
        # --- Patch.4 ---
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    elif get_parallel_state().fsdp_enabled:
        # --- Patch.5 ---
        # Modification: add dummy ViT forward to avoid FSDP reduce-scatter hang
        # when some ranks get None pixel_values_videos while others get valid pixel_values_videos
        fake_embeds, fake_deepstack = self.visual.dummy_forward()
        fake_embeds = fake_embeds.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.5 ---

    # Prepare sliced masks for deepstack use
    rank_image_mask = None
    rank_video_mask = None

    # --- Patch.6 ---
    # Modification: sequence parallel patch
    if self.training and get_parallel_state().sp_enabled:
        # (batch_size, seq_len, hidden_size // sp_size) back to (batch_size, seq_len // sp_size, hidden_size)
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )

        # After sequence is scattered, do all_gather on deepstack embeddings
        # and use masks to select the corresponding visual tokens for this rank
        sp_size = get_parallel_state().sp_size
        sp_rank = get_parallel_state().sp_rank

        if pixel_values is not None:
            # Do all_gather on deepstack_image_embeds
            # (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size)
            deepstack_image_embeds = [
                _Gather.apply(get_parallel_state().sp_group, embed, 0, False) for embed in deepstack_image_embeds
            ]

            # Now use image_mask to select visual tokens for this rank's sequence slice
            # image_mask is (batch_size, seq_len, hidden_size // sp_size) before gather_heads_scatter_seq
            image_mask_1d = image_mask[..., 0]  # (batch_size, seq_len)

            # Determine which sequence positions belong to this rank
            seq_len = image_mask_1d.shape[1]
            seq_per_rank = seq_len // sp_size
            rank_start = sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            # Get the mask for this rank's sequence slice and save it for later
            rank_image_mask = image_mask_1d[:, rank_start:rank_end]  # (batch_size, seq_len // sp_size)
            # Count how many visual tokens are before this rank's slice
            before_rank_mask = image_mask_1d[:, :rank_start]
            offset = before_rank_mask.sum().item()
            # Count how many visual tokens are in this rank's slice
            num_visual_tokens = rank_image_mask.sum().item()
            # Slice the all-gathered deepstack embeddings
            deepstack_image_embeds = [embed[offset : offset + num_visual_tokens] for embed in deepstack_image_embeds]

        if pixel_values_videos is not None:
            # Do all_gather on deepstack_video_embeds
            # (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size)
            deepstack_video_embeds = [
                _Gather.apply(get_parallel_state().sp_group, embed, 0, False) for embed in deepstack_video_embeds
            ]

            # Same logic for video embeddings
            video_mask_1d = video_mask[..., 0]  # (batch_size, seq_len)
            seq_len = video_mask_1d.shape[1]
            seq_per_rank = seq_len // sp_size
            rank_start = sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            # Get the mask for this rank's sequence slice and save it for later
            rank_video_mask = video_mask_1d[:, rank_start:rank_end]
            before_rank_mask = video_mask_1d[:, :rank_start]
            offset = before_rank_mask.sum().item()
            num_visual_tokens = rank_video_mask.sum().item()
            deepstack_video_embeds = [embed[offset : offset + num_visual_tokens] for embed in deepstack_video_embeds]
    # --- Patch.6 ---

    visual_pos_masks = None
    deepstack_visual_embeds = None

    # --- Patch.7 ---
    # Modification: use pixel_values and pixel_values_videos instead of masks
    if pixel_values is not None and pixel_values_videos is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        # reuse the sliced masks if SP is enabled
        if rank_image_mask is not None:
            image_mask = rank_image_mask
        else:
            image_mask = image_mask[..., 0]
        if rank_video_mask is not None:
            video_mask = rank_video_mask
        else:
            video_mask = video_mask[..., 0]
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
        # Modification: Reuse the sliced mask if SP is enabled
        if rank_image_mask is not None:
            image_mask = rank_image_mask
        else:
            image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif pixel_values_videos is not None:
        # Modification: Reuse the sliced mask if SP is enabled
        if rank_video_mask is not None:
            video_mask = rank_video_mask
        else:
            video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds
    else:
        # Both pixel_values and pixel_values_videos are None
        # still pass fake_deepstack to language_model to trigger _deepstack_process
        # to avoid FSDP backward reduce scatter stuck
        # visual_pos_masks remains None, so _deepstack_process will just add 0.0
        if fake_deepstack is not None:
            deepstack_visual_embeds = fake_deepstack
        else:
            deepstack_visual_embeds = None
    # --- Patch.7 ---

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    if attention_mask is not None and position_ids is None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    # --- Patch.9 ---
    # Modification: handle pre-computed position_ids from data preprocessing
    # During training with rmpad_with_pos_ids, position_ids are computed in vlm_data_process.py
    # and stored as (dim=3, L) per sample, then collated to (bs, 3, L) by the data collator.
    # The downstream model expects (3, bs, L), so we transpose here.
    elif position_ids is not None:
        if position_ids.ndim == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()  # bs, 3, l -> 3, bs, l
    # --- Patch.9 ---

    # --- Patch.2 ---
    # Modification: Restore flash attention kwargs for language model to avoid CPU-GPU sync
    kwargs.update(flash_attn_kwargs)
    # --- Patch.2 ---

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        deepstack_visual_embeds=deepstack_visual_embeds,
        visual_pos_masks=visual_pos_masks,
        **kwargs,
    )

    hidden_states = outputs[0]
    # --- Patch.8 ---
    # Modification: sp patch: custom loss computation with Liger kernel and SP loss reduction
    if labels is not None:
        if not get_parallel_state().sp_enabled:
            labels = labels[..., 1:].contiguous()  # shift labels

        labels = labels.view(-1)  # flatten labels

        if is_liger_kernel_available():
            loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="mean")
            if not get_parallel_state().sp_enabled:
                hidden_states = hidden_states[..., :-1, :].contiguous()  # shift hidden states

            hidden_states = hidden_states.view(-1, self.config.text_config.hidden_size)  # flatten hidden states
            loss = loss_fct(self.lm_head.weight, hidden_states, labels)
            logits = None
        else:
            loss_fct = CrossEntropyLoss(reduction="mean")
            logits = self.lm_head(hidden_states)
            if not get_parallel_state().sp_enabled:
                logits = logits[..., :-1, :].contiguous()  # shift logits

            logits = logits.float().view(-1, self.config.text_config.vocab_size)  # flatten logits
            loss = loss_fct(logits, labels)

        if get_parallel_state().sp_enabled:
            num_valid_tokens = (labels != IGNORE_INDEX).sum()
            loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    else:
        logits = self.lm_head(hidden_states)
        loss = None
    # --- Patch.8 ---

    aux_loss = None
    if output_router_logits:
        aux_loss = hf_qwen3_omni_moe.load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    return hf_qwen3_omni_moe.Qwen3OmniMoeThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        aux_loss=aux_loss,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


# ================================================================
# PATCH: Qwen3OmniMoeForConditionalGeneration.__init__
# 1. Simplified for training: only forward thinker, skip talker/code2wav
# ================================================================
def qwen3_omni_moe_for_conditional_generation_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration,
    **kwargs,
) -> Union[tuple, hf_qwen3_omni_moe.Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
    thinker_outputs = self.thinker(
        **kwargs,
    )
    # TODO: talker_outputs
    return thinker_outputs


# ================================================================
# PATCH: Qwen3OmniMoeForConditionalGeneration.__init__
# 1. Propagate _moe_implementation from top-level config to thinker_config
#    and then to text_config before the thinker model is built.
# ================================================================
def qwen3_omni_moe_for_conditional_generation_init(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration,
    config,
):
    # --- Patch.3 ---
    # Propagate _moe_implementation from top-level config to thinker_config
    # and then to text_config before the thinker model is built.
    moe_implementation = getattr(config, "_moe_implementation", "eager")
    config.thinker_config._moe_implementation = moe_implementation
    config.thinker_config.text_config._moe_implementation = moe_implementation
    # --- Patch.3 ---
    super(hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration, self).__init__(config)

    self.thinker = hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration._from_config(config.thinker_config)
    self.has_talker = config.enable_audio_output
    if self.has_talker:
        self.enable_talker()
    self.post_init()


# ================================================================
# NEW: Qwen3OmniMoeThinkerExperts
# Stacked expert weights for fused MoE forward in Qwen3-Omni-MoE thinker layers.
# Used when _moe_implementation == "fused".
# ================================================================
class Qwen3OmniMoeThinkerExperts(nn.Module):
    """Stacked expert weights for fused MoE forward in Qwen3-Omni-MoE thinker layers.

    Stores all expert weights as 3-D tensors (num_experts, out, in) so that
    the fused MoE kernel and Expert Parallelism can operate on them directly.
    The parameter names match the merged-checkpoint convention:
      *.mlp.experts.gate_proj, *.mlp.experts.up_proj, *.mlp.experts.down_proj
    """

    def __init__(self, config) -> None:
        super().__init__()
        from transformers.activations import ACT2FN

        num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
        # Shape convention (same as fused_moe_forward expectation):
        #   gate_proj / up_proj : (num_experts, intermediate_size, hidden_size)
        #   down_proj            : (num_experts, hidden_size,       intermediate_size)
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        return fused_moe_forward(
            module=self,
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj,  # (num_experts, intermediate_size, hidden_size)
            fc1_2_weight=self.up_proj,  # (num_experts, intermediate_size, hidden_size)
            fc2_weight=self.down_proj,  # (num_experts, hidden_size,       intermediate_size)
        )


# ================================================================
# PATCH: Qwen3OmniMoeThinkerTextSparseMoeBlock.__init__ and forward
# 1. Support eager mode: original nn.ModuleList experts (raises error if EP enabled)
# 2. Support fused mode: Qwen3OmniMoeThinkerExperts with stacked weights for fused MoE kernel and EP
# ================================================================
def qwen3_omni_moe_thinker_sparse_moe_block_init(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock,
    config,
) -> None:
    # Call grandparent (nn.Module) init to avoid re-running the original __init__
    nn.Module.__init__(self)
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob
    self._moe_implementation = getattr(config, "_moe_implementation", "eager")

    self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    if self._moe_implementation == "fused":
        # experts sub-module holds stacked weights; its name "experts" ensures
        # parameter FQNs align with the merged checkpoint and parallel_plan.
        self.experts = Qwen3OmniMoeThinkerExperts(config)
    elif self._moe_implementation == "eager":
        self.experts = nn.ModuleList(
            [
                hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
    else:
        raise ValueError(f"Invalid _moe_implementation: {self._moe_implementation!r}. Expected 'eager' or 'fused'.")


def qwen3_omni_moe_thinker_sparse_moe_block_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    if self._moe_implementation == "eager":
        ps = get_parallel_state()
        if ps.ep_enabled:
            raise NotImplementedError(
                "eager MoE does not support Expert Parallelism (EP). Set _moe_implementation='fused' to use EP."
            )
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    else:
        # fused: delegate entirely to Qwen3OmniMoeThinkerExperts
        final_hidden_states = self.experts(hidden_states, routing_weights, selected_experts, self.num_experts)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


# ================================================================
# NEW: _get_parallel_plan
# Provides parallel execution plan for FSDP/SP/EP coordination
# ================================================================
def _get_parallel_plan(self):
    from .parallel_plan import get_parallel_plan

    return get_parallel_plan()


# ================================================================
# apply_veomni_qwen3_omni_moe_patch
# Central entry point to apply all VeOmni patches to HF Qwen3OmniMoe classes
# ================================================================
def apply_veomni_qwen3_omni_moe_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen3_Omni_MoE.")

    # Fix _no_split_modules: use the correct decoder layer class name
    hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModel._no_split_modules = [
        "Qwen3OmniMoeThinkerTextDecoderLayer",
        "Qwen3OmniMoeVisionBlock",
    ]

    # --- Patch.1 ---
    # Patch parallel plan support
    hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModel.get_parallel_plan = _get_parallel_plan
    # --- Patch.1 ---

    # Patch VisionEncoder
    hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder.forward = qwen3_omni_moe_vision_encoder_forward
    hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder.dummy_forward = qwen3_omni_moe_vision_encoder_dummy_forward

    # Patch AudioEncoder
    hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder.forward = qwen3_omni_moe_audio_encoder_forward
    hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder.dummy_forward = qwen3_omni_moe_audio_encoder_dummy_forward

    # Patch ThinkerTextModel
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel.forward = qwen3_omni_moe_thinker_text_model_forward
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel._deepstack_process = (
        qwen3_omni_moe_thinker_text_model_deepstack_process
    )

    # --- Patch.2 ---
    # Patch MoE block (eager/fused modes, EP support)
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock.__init__ = qwen3_omni_moe_thinker_sparse_moe_block_init
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock.forward = qwen3_omni_moe_thinker_sparse_moe_block_forward
    # --- Patch.2 ---

    # Patch ThinkerForConditionalGeneration
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration.__init__ = qwen3_omni_moe_thinker_init
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration.get_position_id_func = (
        qwen3_omni_moe_thinker_get_position_id_func
    )
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration.get_audio_features = (
        qwen3_omni_moe_thinker_get_audio_features
    )
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration.forward = qwen3_omni_moe_thinker_forward

    # Patch ForConditionalGeneration
    hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration.__init__ = qwen3_omni_moe_for_conditional_generation_init
    hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration.forward = qwen3_omni_moe_for_conditional_generation_forward
