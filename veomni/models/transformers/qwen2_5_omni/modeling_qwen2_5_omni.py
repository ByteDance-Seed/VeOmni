# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
import copy
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as hf_qwen25omni
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder as _Qwen2_5OmniAudioEncoder
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration as _Qwen2_5OmniForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    TransformersKwargs,
    Unpack,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration as _Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniVisionEncoder as _Qwen2_5OmniVisionEncoder,
)

from ....data.constants import AUDIO_INPUT_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    pad_tensor,
    slice_input_tensor,
    unpad_tensor,
)
from ....utils import logging
from ..attention_utils import VARLEN_ATTENTION_TYPES


logger = logging.get_logger(__name__)


# ================================================================
# Patch: Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index
# 1. support mixed data of video_w_audio & video_w/o_audio
# 2. set default attention mask to all ones when attention_mask is None, for cleaner code
# ================================================================
def Qwen2_5OmniPreTrainedModelForConditionalGeneration_get_rope_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    # --- Patch.1 ---
    # use_audio_in_video: bool = False,
    # --- Patch.1 ---
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = self.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    audio_token_id = self.config.audio_token_id
    vision_start_token_id = self.config.vision_start_token_id
    audio_start_token_id = self.config.audio_start_token_id
    position_id_per_seconds = self.config.position_id_per_seconds
    seconds_per_chunk = self.config.seconds_per_chunk
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        # --- Patch.2 ---
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        # --- Patch.2 ---
        attention_mask = attention_mask == 1
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        for i, input_ids in enumerate(total_input_ids):
            # --- Patch.2 ---
            input_ids = input_ids[attention_mask[i]]
            # --- Patch.2 ---
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)

            # --- Patch.1 ---
            audio_start_indices = torch.argwhere(input_ids == audio_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(
                input_ids[audio_start_indices - 1] != vision_start_token_id
            )  # audio but not in <video><audio>
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum() + (vision_tokens == audio_start_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + video_nums + audio_nums
            # --- Patch.1 ---

            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video:
                    # --- Patch.1 ---
                    if audio_seqlens[audio_idx] == 0:
                        use_audio_in_video = False
                        audio_idx += 1
                    else:
                        use_audio_in_video = True
                    # --- Patch.1 ---

                    if not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    else:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # --- Patch.2 ---
            position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
            # --- Patch.2 ---
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


# ================================================================
# Patch: Qwen2_5OmniAudioEncoder
# 1. support VARLEN_ATTENTION_TYPES
# 2. input_features shape: (seq_len, dim)
# 3. sp patch
# 4. dummy forward
# ================================================================
class Qwen2_5OmniAudioEncoder(_Qwen2_5OmniAudioEncoder):
    # --- Patch.1 ---
    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    # --- Patch.1 ---

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
        # --- Patch.2 ---
        input_features = input_features.permute(1, 0)  # len, 128 -> 128, len
        # --- Patch.2 ---

        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            unpadded_dim_len = torch.sum(chunk_lengths)
            input_features = gather_outputs(input_features, gather_dim=1, group=get_parallel_state().sp_group)
            sp_padding_size = input_features.size(1) - unpadded_dim_len
            if sp_padding_size > 0:
                input_features = unpad_tensor(input_features, dim=1, padding_size=sp_padding_size)
        # --- Patch.3 ---

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            unpadded_dim_len = cu_seqlens[-1]
            hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
            pad_seq_len = hidden_states.size(0) * get_parallel_state().sp_size - unpadded_dim_len
            if pad_seq_len > 0:
                # Add this extra sequence to cu_seqlens with the padding length
                new_cumsum = cu_seqlens[-1] + pad_seq_len
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        # --- Patch.3 ---

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            hidden_states = gather_outputs(hidden_states, gather_dim=0, group=get_parallel_state().sp_group)
            sp_padding_size = hidden_states.size(0) - unpadded_dim_len
            if sp_padding_size > 0:
                hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
        # --- Patch.3 ---

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            token_audio = slice_input_tensor(token_audio, dim=0, group=get_parallel_state().sp_group)
        # --- Patch.3 ---

        return BaseModelOutput(last_hidden_state=token_audio)

    # --- Patch.4 ---
    def dummy_forward(self):
        if getattr(self, "_dummy_data", None) is None:
            input_features = torch.randn((4, 128), dtype=self.dtype, device=self.device)
            feature_lens = torch.tensor([4], dtype=torch.int64, device=self.device)
            aftercnn_lens = torch.tensor([2], dtype=torch.int64, device=self.device)
            self._dummy_data = {
                "input_features": input_features,
                "feature_lens": feature_lens,
                "aftercnn_lens": aftercnn_lens,
            }
        return self(**self._dummy_data)

    # --- Patch.4 ---


# ================================================================
# Patch: Qwen2_5OmniVisionAttention.forward
# 1. support VARLEN_ATTENTION_TYPES
# 2. use precomputed max_seqlen
# ================================================================
def Qwen2_5OmniVisionAttention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
    key_states = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
    value_states = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
    query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
    key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # --- Patch.1 ---
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
        # --- Patch.1 ---
        # Use cu_seqlens for variable length attention
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
# Patch: Qwen2_5OmniVisionEncoder.forward
# 1. sp patch
# 2. precompute max_seqlen
# 3. dummy forward
# ================================================================
class Qwen2_5OmniVisionEncoder(_Qwen2_5OmniVisionEncoder):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
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

        # sp patch: use all-to-all to get full sequence of hidden_states for window attention
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        unpadded_dim_size = cu_seqlens[-1]
        # --- Patch.1 ---
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

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            if sp_padding_size > 0:
                hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
                rotary_pos_emb = pad_tensor(rotary_pos_emb, dim=0, padding_size=sp_padding_size)
                new_cumsum = cu_seqlens[-1] + sp_padding_size
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
                cu_window_seqlens = torch.cat([cu_window_seqlens, new_cumsum.unsqueeze(0)], dim=0)

            hidden_states = gather_heads_scatter_seq(
                hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
            )
            rotary_pos_emb = slice_input_tensor(rotary_pos_emb, dim=0)
        # --- Patch.1 ---

        # --- Patch.2 ---
        win_max_seqlen = (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().detach().cpu().item()
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
        # --- Patch.2 ---

        # Modification here
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                # --- Patch.2 ---
                max_seqlens_now = max_seqlen
                # --- Patch.2 ---
            else:
                cu_seqlens_now = cu_window_seqlens
                # --- Patch.2 ---
                max_seqlens_now = win_max_seqlen
                # --- Patch.2 ---
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                # --- Patch.2 ---
                max_seqlen=max_seqlens_now,
                # --- Patch.2 ---
                rotary_pos_emb=rotary_pos_emb,
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

        ## --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            if sp_padding_size > 0:
                hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
            hidden_states = gather_heads_scatter_seq(
                hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
            )
        # --- Patch.1 ---

        return hidden_states

    # --- Patch.3 ---
    def dummy_forward(self):
        if getattr(self, "_dummy_data", None) is None:
            pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
            self._dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        return self(**self._dummy_data)

    # --- Patch.3 ---


# ================================================================
# Patch: Qwen2_5OmniThinkerForConditionalGeneration
# 1. precompute position id for veomni
# 2. support mixed data of video_w_audio & video_w/o_audio (filter audio_feature_lengths)
# veomni precompute audio_feature_lengths from feature_attention_mask
# 3. sp patch
# 4. use veomni precomputed multimodal masks
# 5. dummy forward
# 6. transpose precomputed position ids
# 7. fused loss function
# ================================================================


# --- Patch.1 ---
def get_position_id(main_func, self, **kwargs):
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, 1, l), rope_deltas (1, 1)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}


# --- Patch.1 ---


class Qwen2_5OmniThinkerForConditionalGeneration(_Qwen2_5OmniThinkerForConditionalGeneration):
    # --- Patch.1 ---
    def get_position_id_func(self):
        fake_config = copy.copy(self.config)
        fake_config.image_token_id = IMAGE_INPUT_INDEX
        fake_config.video_token_id = VIDEO_INPUT_INDEX
        fake_config.audio_token_id = AUDIO_INPUT_INDEX
        fake_model = SimpleNamespace(
            config=fake_config,
            spatial_merge_size=self.spatial_merge_size,
            get_llm_pos_ids_for_vision=partial(
                Qwen2_5OmniThinkerForConditionalGeneration.get_llm_pos_ids_for_vision, None
            ),
            get_chunked_index=partial(Qwen2_5OmniThinkerForConditionalGeneration.get_chunked_index, None),
        )
        return partial(get_position_id, Qwen2_5OmniThinkerForConditionalGeneration.get_rope_index, fake_model)

    # --- Patch.1 ---

    # --- Patch.2 ---
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths
        )
        # --- Patch.2 ---
        feature_lens = audio_feature_lengths
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state

        # --- Patch.3 ---
        # audio features is sp_sliced feature
        # if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
        #     raise ValueError("length of audio_features should match audio_output_lengths")
        # --- Patch.3 ---
        return audio_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # --- Patch.2 ---
        # feature_attention_mask: Optional[torch.Tensor] = None,
        # --- Patch.2 ---
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # --- Patch.2 ---
        # use_audio_in_video: Optional[bool] = None,
        # --- Patch.2 ---
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        # --- Patch.4 ---
        image_mask: Optional[torch.BoolTensor] = None,
        video_mask: Optional[torch.BoolTensor] = None,
        audio_mask: Optional[torch.BoolTensor] = None,
        # --- Patch.4 ---
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            if cache_position is not None and cache_position[0] == 0:
                # --- Patch.4 ---
                input_ids[image_mask | audio_mask | video_mask] = 0
                # --- Patch.4 ---
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # --- Patch.3 ---
            if get_parallel_state().sp_enabled:
                inputs_embeds = gather_seq_scatter_heads(
                    inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
                )
            # --- Patch.3 ---

            # 2. Merge text , audios , image and video
            # process audio features

            # --- Patch.2 ---
            if input_features is not None:
                valid_mask = audio_feature_lengths != 0
                # filter videos without audios, the origin invalid audio_feature_lengths only used for get_rope_index, now filter them out
                audio_feature_lengths = audio_feature_lengths[valid_mask]

                if (
                    input_features.shape[0] == 0
                ):  # input_features is (0, dim) when no audio in all videos, we do not forward audio_tower
                    input_features = None
            # --- Patch.2 ---

            if input_features is not None:
                audio_features = self.get_audio_features(
                    input_features,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

                # --- Patch.3 ---
                if get_parallel_state().sp_enabled:
                    # audio_features gathered in audio_tower
                    audio_features = gather_seq_scatter_heads(
                        audio_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                # --- Patch.3 ---

                # --- Patch.4 ---
                audio_features = audio_features[: audio_mask.sum()]
                audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                # --- Patch.4 ---

            # --- Patch.5 ---
            elif get_parallel_state().fsdp_enabled:
                fake_embeds = self.audio_tower.dummy_forward().last_hidden_state.mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.5 ---

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # --- Patch.3 ---
                if get_parallel_state().sp_enabled:
                    image_embeds = gather_seq_scatter_heads(
                        image_embeds, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                # --- Patch.3 ---
                # --- Patch.4 ---
                image_embeds = image_embeds[: image_mask.sum()]
                image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                # --- Patch.4 ---

            # --- Patch.5 ---
            elif get_parallel_state().fsdp_enabled:
                fake_embeds = self.visual.dummy_forward().mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.5 ---

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # --- Patch.3 ---
                if get_parallel_state().sp_enabled:
                    video_embeds = gather_seq_scatter_heads(
                        video_embeds, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                # --- Patch.3 ---

                # --- Patch.4 ---
                video_embeds = video_embeds[: video_mask.sum()]
                video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                # --- Patch.4 ---
            # --- Patch.5 ---
            elif get_parallel_state().fsdp_enabled:
                fake_embeds = self.visual.dummy_forward().mean() * 0.0
                fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.5 ---

            # --- Patch.3 ---
            if get_parallel_state().sp_enabled:
                inputs_embeds = gather_heads_scatter_seq(
                    inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
                )
            # --- Patch.3 ---

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
                    # --- Patch.2 ---
                    # use_audio_in_video,
                    # --- Patch.2 ---
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
        # --- Patch.6 ---
        elif position_ids is not None:
            if position_ids.shape[1] == 3:
                position_ids = position_ids.transpose(0, 1).contiguous()  # bs, dim, l -> dim, bs, l
        # --- Patch.6 ---

        # --- Patch.3 ---
        if get_parallel_state().sp_enabled:
            position_ids = slice_input_tensor(position_ids, dim=-1)
        # --- Patch.3 ---

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        # --- Patch.7 ---
        loss = None
        logits = None

        if labels is not None:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
        # --- Patch.7 ---

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


# ================================================================
# Patch: Qwen2_5OmniForConditionalGeneration
# 1. support mixed data of video_w_audio & video_w/o_audio
# 2. use veomni precomputed multimodal masks
# 3. forward function for training
# 4. fix _no_split_modules
# ================================================================
class Qwen2_5OmniForConditionalGeneration(_Qwen2_5OmniForConditionalGeneration):
    # --- Patch.4 ---
    _no_split_modules = ["Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock", "Qwen2_5OmniAudioEncoderLayer"]
    # --- Patch.4 ---

    @torch.no_grad()
    # TODO: raushan, defaults should be saved in generation config
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        # --- Patch.1 ---
        # use_audio_in_video: bool = False,
        # --- Patch.1 ---
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        if speaker not in self.speaker_map:
            raise ValueError(f"{speaker} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        # --- Patch.1 ---
        shared_kwargs = {}
        # --- Patch.1 ---

        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # --- Patch.2 ---
            elif key == "image_mask" or key == "audio_mask" or key == "video_mask":
                thinker_kwargs[key] = value
            # --- Patch.2 ---
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[speaker]

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        # 2. Generate speech tokens from talker module
        # --- Patch.2 ---
        embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(input_ids.device)
        if thinker_kwargs.get("input_features") is not None:
            audio_mask = kwargs["audio_mask"]
            audio_mask_tensor = torch.zeros(
                [audio_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=input_ids.device,
            )
            audio_mask = audio_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
        if thinker_kwargs.get("pixel_values") is not None:
            image_mask = kwargs["image_mask"]
            image_mask_tensor = torch.zeros(
                [image_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=input_ids.device,
            )
            image_mask = image_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
        if thinker_kwargs.get("pixel_values_videos") is not None:
            video_mask = kwargs["video_mask"]
            video_mask_tensor = torch.zeros(
                [video_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=input_ids.device,
            )
            video_mask = video_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)
        # --- Patch.2 ---

        processed_thinker_hidden = (
            (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
        ) + thinker_result.hidden_states[1:]
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(input_ids.device)
        thinker_token_embeds = [
            token_hidden_states[0].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
        ]
        thinker_hidden_states = [
            token_hidden_states[-1].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
        ]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids,
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=input_ids.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=input_ids.device),
            ],
            dim=1,
        )

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device)
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(input_ids.device)
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_token = torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=input_ids.device)
        eos_embedding = thinker_embed_tokens(eos_token).to(input_ids.device)

        pad_token = torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=input_ids.device)
        pad_embedding = thinker_embed_tokens(pad_token).to(input_ids.device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        talker_attention_mask = None
        if "attention_mask" in kwargs:
            talker_attention_mask = torch.cat(
                [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
            ).to(input_ids.device)

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(input_ids.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

        # 3. Generate wavs from code
        if self.token2wav.dtype != torch.float:
            self.token2wav.float()

        wav = self.token2wav(
            talker_generate_codes.to(input_ids.device),
            conditioning=speaker_params["cond"].to(input_ids.device).float(),
            reference_mel=speaker_params["ref_mel"].to(input_ids.device).float(),
            **token2wav_kwargs,
        )

        return thinker_result.sequences, wav.float()

    # --- Patch.3 ---
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        image_mask: Optional[torch.BoolTensor] = None,
        video_mask: Optional[torch.BoolTensor] = None,
        audio_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        thinker_outputs = self.thinker(
            input_ids=input_ids,
            input_features=input_features,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            video_second_per_grid=video_second_per_grid,
            image_mask=image_mask,
            video_mask=video_mask,
            audio_mask=audio_mask,
            **kwargs,
        )
        # TODO: talker_outputs
        return thinker_outputs

    # --- Patch.3 ---


def apply_veomni_qwen25omni_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen2.5_Omni.")

    hf_qwen25omni.Qwen2_5OmniForConditionalGeneration = Qwen2_5OmniForConditionalGeneration
    hf_qwen25omni.Qwen2_5OmniThinkerForConditionalGeneration = Qwen2_5OmniThinkerForConditionalGeneration
    hf_qwen25omni.Qwen2_5OmniVisionEncoder = Qwen2_5OmniVisionEncoder
    hf_qwen25omni.Qwen2_5OmniAudioEncoder = Qwen2_5OmniAudioEncoder
    hf_qwen25omni.Qwen2_5OmniVisionAttention.forward = Qwen2_5OmniVisionAttention_forward
    Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index = (
        Qwen2_5OmniPreTrainedModelForConditionalGeneration_get_rope_index
    )
