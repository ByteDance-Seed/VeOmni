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
import inspect
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .....utils.constants import AUDIO_INPUT_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....transformers.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)
from ..base import BaseFoundationModelMixin
from .configuration_qwen2_5_omni_foundation import Qwen25OmniFoundationModelConfig


def parse_position_id_kwargs(input_ids: torch.Tensor, attention_mask: torch.Tensor, grid_thw: Dict = {}, **kwargs):
    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "image" in grid_thw:
        return_dict["image_grid_thw"] = grid_thw["image"]
    if "video" in grid_thw:
        return_dict["video_grid_thw"] = grid_thw["video"]
        return_dict["second_per_grids"] = torch.tensor([1.0] * len(grid_thw["video"])).to(input_ids.device)
    if "feature_lengths" in kwargs:  # audio 'grid_thw'
        return_dict["audio_seqlens"] = kwargs["feature_lengths"]["audio"]
    return return_dict


class Qwen25OmniFoundationModel(BaseFoundationModelMixin, Qwen2_5OmniThinkerForConditionalGeneration, GenerationMixin):
    config_class = Qwen25OmniFoundationModelConfig
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]
    forward_valid_kwargs = list(
        inspect.signature(Qwen2_5OmniThinkerForConditionalGeneration.forward).parameters.keys()
    )

    def __init__(self, config: Qwen25OmniFoundationModelConfig, **kwargs):
        BaseFoundationModelMixin.__init__(self, config, **kwargs)
        Qwen2_5OmniPreTrainedModelForConditionalGeneration.__init__(self, config, **kwargs)
        GenerationMixin.__init__(self)
        self.config = config
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None

        self.image_token_index = IMAGE_INPUT_INDEX
        self.video_token_index = VIDEO_INPUT_INDEX
        self.audio_token_index = AUDIO_INPUT_INDEX
        self.post_init()

    def get_position_id_func(self):
        return [parse_position_id_kwargs, Qwen2_5OmniThinkerForConditionalGeneration.get_position_id_func(self)]

    def prepare_inputs_for_generation(
        self,
        input_ids,
        rope_deltas=None,
        **kwargs,
    ):
        if rope_deltas is not None:
            self.rope_deltas = rope_deltas
        return super().prepare_inputs_for_generation(
            input_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        new_kwargs = {k: v for k, v in kwargs.items() if k in self.forward_valid_kwargs}
        return super().forward(input_ids, **new_kwargs)
