from typing import Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ....transformers.llama.generated.patched_modeling_llama_gpu import LlamaForCausalLM
from ..base import BaseFoundationModelMixin
from .configuration_llama_foundation import LlamaFoundationConfig


class LlamaFoundationModel(BaseFoundationModelMixin, LlamaForCausalLM):
    config_class = LlamaFoundationConfig

    def __init__(self, config: LlamaFoundationConfig, **kwargs):
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
