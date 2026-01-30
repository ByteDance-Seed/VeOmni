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

from typing import (
    Dict,
)

import torch
import torch.nn as nn

from .....utils import logging
from ....seed_omni.projector import build_feature_projector
from ....transformers.janus.modeling_janus import CLIPVisionTower, MlpProjector
from ..base import BaseEncoderModelMixin
from .configuration_janus_siglip import JanusSigLIPEncoderConfig


logger = logging.get_logger(__name__)


class JanusSiglipEncoder(BaseEncoderModelMixin, CLIPVisionTower):
    config_class = JanusSigLIPEncoderConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: JanusSigLIPEncoderConfig):
        super().__init__(config)
        self.config = config

        self.aligner = MlpProjector(
            depth=config.aligner_depth,
            input_dim=config.aligner_input_dim,
            n_embed=config.n_embed,
            projector_type=config.aligner_projector_type,
        )

        if config.add_projector and config.output_size is not None:
            self.projector = build_feature_projector(config.n_embed, config.output_size)
        else:
            if config.output_size is not None and config.output_size != config.n_embed:
                raise ValueError("`output_size` should be same as `hidden_size`.")
            self.projector = nn.Identity()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector and self.config.output_size is not None:
            self.projector.requires_grad_(True)
        else:
            self.aligner.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, **kwargs):
        image_features = super().forward(features)
        return self.projector(self.aligner(image_features))

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn((1, 3, 384, 384), dtype=self.dtype, device=self.device)
        return {"features": pixel_values}
