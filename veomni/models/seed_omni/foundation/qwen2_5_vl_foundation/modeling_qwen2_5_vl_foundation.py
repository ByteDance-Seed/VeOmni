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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .....utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....transformers.qwen2_5vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)
from ..base import BaseFoundationModelMixin
from .configuration_qwen2_5_vl_foundation import Qwen25VLFoundationConfig


def parse_position_id_kwargs(input_ids: torch.Tensor, attention_mask: torch.Tensor, grid_thw: Dict = {}, **kwargs):
    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "image" in grid_thw:
        return_dict["image_grid_thw"] = grid_thw["image"]
    if "video" in grid_thw:
        return_dict["video_grid_thw"] = grid_thw["video"]
    return return_dict


class Qwen25VLFoundationModel(BaseFoundationModelMixin, Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen25VLFoundationConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config: Qwen25VLFoundationConfig, **kwargs):
        BaseFoundationModelMixin.__init__(self, config, **kwargs)
        Qwen2_5_VLPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Overwrite token ids
        self.image_token_id = IMAGE_INPUT_INDEX
        self.video_token_id = VIDEO_INPUT_INDEX

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_id_func(self):
        return [parse_position_id_kwargs, Qwen2_5_VLForConditionalGeneration.get_position_id_func(self)]

    def prepare_inputs_for_generation(
        self,
        input_ids,
        rope_deltas=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_mask=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        if rope_deltas is not None:
            self.rope_deltas = rope_deltas
        return super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            position_ids,
            use_cache,
            pixel_values,
            pixel_values_videos,
            image_mask,
            image_grid_thw,
            video_grid_thw,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
