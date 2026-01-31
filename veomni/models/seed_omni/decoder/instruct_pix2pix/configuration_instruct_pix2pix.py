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

from typing import Dict

from transformers import CONFIG_MAPPING, PretrainedConfig

from .....utils import logging


logger = logging.get_logger(__name__)


def dict_to_config(config_dict: Dict) -> PretrainedConfig:
    return CONFIG_MAPPING[config_dict["model_type"]].from_dict(config_dict)


class InstructPix2PixConfig(PretrainedConfig):
    model_type = "instruct_pix2pix"

    def __init__(
        self,
        hidden_size=None,
        condition_dim=2048,
        unet_config=None,
        vae_config=None,
        scheduler_config=None,
        force_zeros_for_empty_prompt=True,
        is_cosxl_edit=False,
        **kwargs,
    ):
        """
        hidden_size:
            dim for hidden_states from LLM.
        condition_dim:
            condition_dim for prompt_embeds input as condition.
            The condition is injected to the model through cross-attention, or transformed to a patch-level embeds and concat to latent noises.
            2048 is the default condition_dim for diffusers/sdxl-instructpix2pix-768.
            In future, condition_dim can be get from model.unet.up_blocks[0].attentions[0].transformer_blocks[0].attn2.to_k
        """
        self.hidden_size = hidden_size
        self.condition_dim = condition_dim
        self.unet_config = unet_config
        self.vae_config = vae_config
        self.scheduler_config = scheduler_config
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt
        self.is_cosxl_edit = is_cosxl_edit
        super().__init__(**kwargs)
