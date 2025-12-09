# This file is based on code from https://github.com/ai-forever/MoVQGAN
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

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import CrossEntropyLoss

from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import reduce_sequence_parallel_loss
from ....seed_omni.projector import build_feature_projector, build_vit_decoder
from ....transformers.movqgan.modeling_movqgan import MoVQGAN
from ..base import BaseDecoderModelMixin, BaseDecoderOutput
from .configuration_movqgan import MoVQGANDecoderConfig


@dataclass
class MoVQGANDecoderOutput(BaseDecoderOutput):
    losses: Optional[Dict] = None


class GenerationHead(nn.Module):
    def __init__(self, n_embed, image_token_embed):
        super().__init__()
        self.output_mlp_projector = nn.Linear(n_embed, n_embed)
        self.vision_activation = nn.GELU()
        self.vision_head = nn.Linear(n_embed, image_token_embed)

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


class MoVQGANDecoder(BaseDecoderModelMixin, MoVQGAN):
    config_class = MoVQGANDecoderConfig
    _no_split_modules = ["ViTLayer"]

    def __init__(self, config: MoVQGANDecoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if config.use_semantic_decoder:
            self.semantic_decoder = build_vit_decoder(
                input_dim=config.z_channels,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                output_dim=config.semantic_dim,
            )
        if config.add_projector and config.output_size is not None:
            if config.use_semantic_decoder:
                self.gen_aligner = build_feature_projector(config.semantic_dim, config.output_size)
            else:
                self.gen_aligner = build_feature_projector(config.embed_dim, config.output_size)
            self.gen_head = GenerationHead(config.output_size, config.n_embed)
        else:
            if config.output_size is not None and config.output_size != config.n_embed:
                raise ValueError("`output_size` should be same as `hidden_size`.")
            self.gen_aligner = nn.Identity()
            self.gen_head = nn.Identity()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.add_projector and self.config.output_size is not None:
            self.gen_aligner.requires_grad_(True)
            self.gen_head.requires_grad_(True)
        if not self.config.freeze_codebook:
            self.quantize.embedding.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, **kwargs):
        bs = features.shape[0]
        _, _, info = self.encode(features)
        indices = info[-1]
        token_num = indices.shape[0] // bs
        assert token_num in [1, 1024]  # 1 for fake input
        img_size = int(math.sqrt(token_num))
        indices = rearrange(indices, "(b d) -> b d", b=bs, d=token_num)
        embeds = self.quantize.embedding(indices)
        if self.config.use_semantic_decoder:
            embeds = embeds.permute(0, 2, 1).reshape(bs, -1, img_size, img_size)
            embeds = self.semantic_decoder(embeds)
        embeds = self.gen_aligner(embeds)
        embeds = rearrange(embeds, "b d c -> (b d) c", b=bs, d=token_num)
        indices = rearrange(indices, "b d -> (b d)", b=bs, d=token_num)
        return embeds, indices

    def _get_lm_dummy_data(self):
        features = torch.randn((1, 3, 256, 256), dtype=self.dtype, device=self.device)
        return {"features": features}

    def embed_to_indice(
        self, hidden_states: torch.Tensor, temperature: float = 1.0, cfg: bool = False, cfg_weight: int = 5
    ):
        logits = self.gen_head(hidden_states)
        if cfg:
            logits_shape = logits.shape
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            logits = torch.cat([logits, logits], dim=1).view(*logits_shape)
        probs = torch.softmax(logits / temperature, dim=-1)
        bs, dim = probs.shape[0], probs.shape[-1]
        probs = probs.reshape(-1, dim)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(bs, -1)
        return logits, next_token

    def lm_head(self, hidden_states: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        logits = self.gen_head(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            if get_parallel_state().sp_enabled:
                num_valid_tokens = (labels != -100).sum()
                loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)

        return MoVQGANDecoderOutput(
            loss=loss,
            logits=logits,
        )

    def lm_embed(self, hidden_states: torch.Tensor, **kwargs):
        logits, indices = self.embed_to_indice(hidden_states, **kwargs)
        embeds = self.gen_aligner(self.quantize.embedding(indices))
        return embeds

    def lm_generate(self, indices: torch.Tensor, **kwargs):
        indices = indices.reshape(-1, 32, 32)
        bs = indices.shape[0]
        return self.decode_code(indices, shape=[bs, 32, 32, 4])

    def forward(self, features: torch.Tensor, **kwargs):
        embeds, loss, _ = self.encode(features)
        loss_dict = {"commit_loss": loss.item()}
        if self.config.use_semantic_decoder and "teacher_feature" in kwargs:
            gt = kwargs["teacher_feature"]
            pred = self.semantic_decoder(embeds)
            gt = gt / gt.norm(dim=-1, keepdim=True)
            pred = pred / pred.norm(dim=-1, keepdim=True)
            sem_loss = (1 - (gt * pred).sum(-1)).mean()
            loss += sem_loss
            loss_dict["sem_loss"] = sem_loss.item()

        rec = self.decode(embeds)
        reconstruction_loss_fn = nn.MSELoss()
        rec_loss = reconstruction_loss_fn(features, rec)
        loss += rec_loss
        loss_dict["rec_loss"] = rec_loss.item()
        return MoVQGANDecoderOutput(
            loss=loss,
            logits=rec,
            losses=loss_dict,
        )
