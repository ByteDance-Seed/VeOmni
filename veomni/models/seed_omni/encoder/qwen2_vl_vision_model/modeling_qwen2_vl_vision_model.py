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
import torch.nn.functional as F

from ....transformers.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from ...projector import build_feature_projector
from ..base import BaseEncoderModelMixin
from .configuration_qwen2_vl_vision_model import Qwen2VLVisionModelConfig


class Qwen2VLVisionModel(BaseEncoderModelMixin, Qwen2VisionTransformerPretrainedModel):
    config_class = Qwen2VLVisionModelConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config: Qwen2VLVisionModelConfig):
        super().__init__(config)
        self.config = config
        if config.add_projector and config.output_size is not None:
            self.projector = build_feature_projector(config.hidden_size, config.output_size)
        else:
            if config.output_size and config.output_size != config.hidden_size:
                raise ValueError("`output_size` should be same as `hidden_size`.")

            self.projector = nn.Identity()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector:
            self.projector.requires_grad_(True)
        else:
            self.merger.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(features)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=dtype
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        if self.config.return_hidden_states:
            return hidden_states
        hidden_states = self.projector(self.merger(hidden_states))
        return hidden_states

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
        return {"features": pixel_values, "grid_thw": grid_thw}
