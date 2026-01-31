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

from transformers import PretrainedConfig

from .....utils import logging


logger = logging.get_logger(__name__)


class UltraEditConfig(PretrainedConfig):
    model_type = "ultra_edit"

    def __init__(
        self,
        output_size=None,
        condition_dim=2048,
        transformer_config=None,
        vae_config=None,
        scheduler_config=None,
        add_projector=False,
        patch_size1=14,  # align with qwen2vl,
        patch_size2=2,  # upsample to [16,16]
        proj_hidden_dim=1280,  # align with qwen2vl
        **kwargs,
    ):
        """
        output_size:
            dim for hidden_states from LLM.
        condition_dim:
            condition_dim for prompt_embeds input as condition.
            The condition is injected to the model through cross-attention, or transformed to a patch-level embeds and concat to latent noises.
            2048 is the default condition_dim for stablediffusion3instructpix2pixpipeline.
            In future, condition_dim can be get from model.transformer
        """
        self.output_size = output_size
        self.condition_dim = condition_dim
        self.transformer_config = transformer_config
        self.vae_config = vae_config
        self.scheduler_config = scheduler_config
        self.add_projector = add_projector
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2
        self.proj_hidden_dim = proj_hidden_dim
        super().__init__(**kwargs)
