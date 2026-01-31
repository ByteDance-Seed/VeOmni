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

import torch
import torch.nn as nn

from ....transformers.qwen2_5vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from ...projector import build_feature_projector
from ..base import BaseEncoderModelMixin
from .configuration_qwen2_5_vl_vision_model import Qwen25VLVisionModelConfig


class Qwen25VLVisionModel(BaseEncoderModelMixin, Qwen2_5_VisionTransformerPretrainedModel):
    config_class = Qwen25VLVisionModelConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config: Qwen25VLVisionModelConfig):
        super().__init__(config)
        self.config = config
        if config.add_projector and config.output_size is not None:
            self.projector = build_feature_projector(config.out_hidden_size, config.output_size)
        else:
            if config.output_size and config.output_size != config.out_hidden_size:
                raise ValueError("`output_size` should be same as `out_hidden_size`.")

            self.projector = nn.Identity()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector and self.config.output_size is not None:
            self.projector.requires_grad_(True)
            if self.config.train_origin_projector:
                self.merger.requires_grad_(True)
        else:
            self.merger.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(features, grid_thw)

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
        return {"features": pixel_values, "grid_thw": grid_thw}
