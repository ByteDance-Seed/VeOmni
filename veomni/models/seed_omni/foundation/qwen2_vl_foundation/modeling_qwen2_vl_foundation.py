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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .....data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....transformers.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
)
from ..base import BaseFoundationModelMixin
from .configuration_qwen2_vl_foundation import Qwen2VLFoundationConfig


class Qwen2VLFoundationModel(BaseFoundationModelMixin, Qwen2VLForConditionalGeneration):
    config_class = Qwen2VLFoundationConfig

    def __init__(self, config: Qwen2VLFoundationConfig, **kwargs):
        BaseFoundationModelMixin.__init__(self, config, **kwargs)
        Qwen2VLPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Overwrite token ids
        self.config.image_token_id = IMAGE_INPUT_INDEX
        self.config.video_token_id = VIDEO_INPUT_INDEX

        # Initialize weights and apply final processing
        self.post_init()

    def get_generation_position_id(self, image_grid_thw: torch.Tensor):
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        fake_input_ids = (
            torch.tensor([vision_start_token_id] + [image_token_id] * (image_grid_thw[1] * image_grid_thw[2]))
            .unsqueeze(0)
            .to(self.device)
        )
        attention_mask = torch.ones_like(fake_input_ids).to(self.device)
        image_grid_thw[1] *= self.config.vision_config.spatial_merge_size
        image_grid_thw[2] *= self.config.vision_config.spatial_merge_size
        position_ids, rope_deltas = self.get_rope_index(
            fake_input_ids, image_grid_thw.unsqueeze(0).to(self.device), attention_mask
        )
        # shift fake vision_start_token
        position_ids = position_ids[:, :, 1:] - 1
        position_ids += self.rope_deltas
        self.rope_deltas += rope_deltas
        return position_ids

    def get_position_id_func(self):
        return Qwen2VLForConditionalGeneration.get_position_id_func(self)

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
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
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
