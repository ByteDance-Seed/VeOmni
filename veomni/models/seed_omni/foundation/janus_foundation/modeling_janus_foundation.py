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

from typing import Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ....transformers.llama.modeling_llama import LlamaForCausalLM
from ..base import BaseFoundationModelMixin
from .configuration_janus_foundation import JanusFoundationConfig


class JanusFoundationModel(BaseFoundationModelMixin, LlamaForCausalLM):
    config_class = JanusFoundationConfig

    def __init__(self, config: JanusFoundationConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        position_ids: torch.LongTensor = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids.squeeze(1)  # bs, 1, l -> bs, l
        if inputs_embeds is not None:
            return super().forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                **kwargs,
            )
        else:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                **kwargs,
            )
