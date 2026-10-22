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

"""Config for :class:`LladaUniImageTokenizer` (GlmImage SigLIP-ViT + VQ tokenizer).

Wraps the GlmImage vision tower (``GlmImageVisionConfig``) and VQ codec
(``GlmImageVQVAEConfig``). The tokenizer maps an image to discrete VQ token ids
in ``[0, num_embeddings)`` which are then offset by ``image_token_offset`` into
the unified LLaDA2.0-Uni vocabulary (``[157184, 173567]`` for the 16384 codes).
"""

from typing import Any, Dict, Optional

from transformers.models.glm_image.configuration_glm_image import (
    GlmImageVisionConfig,
    GlmImageVQVAEConfig,
)

from ...module_configuration_base import OmniModuleConfig


class LladaUniImageTokenizerConfig(OmniModuleConfig):
    """Config for the LLaDA2.0-Uni discrete image tokenizer (vision + VQ only)."""

    model_type = "llada2_uni_image_tokenizer"
    sub_configs = {"vision_config": GlmImageVisionConfig, "vq_config": GlmImageVQVAEConfig}

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any] | GlmImageVisionConfig] = None,
        vq_config: Optional[Dict[str, Any] | GlmImageVQVAEConfig] = None,
        image_token_offset: int = 157184,
        image_start_token_id: int = 156901,
        image_end_token_id: int = 156902,
        boi_token_id: int = 156904,
        reserved_token_base: int = 156895,
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            self.vision_config = GlmImageVisionConfig()
        elif isinstance(vision_config, GlmImageVisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = GlmImageVisionConfig(**vision_config)

        if vq_config is None:
            self.vq_config = GlmImageVQVAEConfig()
        elif isinstance(vq_config, GlmImageVQVAEConfig):
            self.vq_config = vq_config
        else:
            self.vq_config = GlmImageVQVAEConfig(**vq_config)

        self.image_token_offset = image_token_offset
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.boi_token_id = boi_token_id
        self.reserved_token_base = reserved_token_base
        self.freeze = freeze


__all__ = ["LladaUniImageTokenizerConfig"]
