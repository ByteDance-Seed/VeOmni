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

from ....transformers.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
from ...projector import build_feature_projector
from ..base import BaseEncoderModelMixin
from .configuration_qwen2_5_omni_audio_model import Qwen25OmniAudioModelConfig


class Qwen25OmniAudioModel(BaseEncoderModelMixin, Qwen2_5OmniAudioEncoder):
    config_class = Qwen25OmniAudioModelConfig
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]

    def __init__(self, config: Qwen25OmniAudioModelConfig):
        super().__init__(config)
        self.config = config
        if config.add_projector and config.output_size is not None:
            self.projector = build_feature_projector(config.output_dim, config.output_size)
        else:
            if config.output_size and config.output_size != config.output_dim:
                raise ValueError("`output_size` should be same as `output_dim`.")

            self.projector = nn.Identity()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector and self.config.output_size is not None:
            self.projector.requires_grad_(True)
            if self.config.train_origin_projector:
                self.proj.requires_grad_(True)
        else:
            self.proj.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, feature_lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        valid_mask = feature_lengths > 0
        feature_lengths = feature_lengths[valid_mask]
        audio_feat_lengths, _ = self._get_feat_extract_output_lengths(feature_lengths)
        hidden_states = (
            super().forward(features, feature_lens=feature_lengths, aftercnn_lens=audio_feat_lengths).last_hidden_state
        )
        return self.projector(hidden_states)

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        features = torch.randn((4, 128), dtype=self.dtype, device=self.device)
        feature_lens = torch.tensor([[4]], dtype=torch.int64, device=self.device)
        return {"features": features, "feature_lengths": feature_lens}
