from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import CrossEntropyLoss

from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import reduce_sequence_parallel_loss
from ....seed_omni.projector import build_feature_projector
from ....transformers.janus.modeling_janus import MlpProjector, VQModel, vision_head
from ..base import BaseDecoderModelMixin, BaseDecoderOutput
from .configuration_janusvq16 import JanusVQ16DecoderConfig


def build_projector(config: JanusVQ16DecoderConfig, n_embed):
    gen_aligner = MlpProjector(
        depth=config.gen_aligner_depth,
        input_dim=config.gen_aligner_input_dim,
        n_embed=n_embed,
        projector_type=config.gen_aligner_projector_type,
    )
    gen_head = vision_head(
        n_embed=n_embed,
        image_token_embed=config.gen_head_embed,
        image_token_size=config.codebook_size,
    )
    gen_embed = torch.nn.Embedding(config.codebook_size, config.codebook_embed_dim)
    return gen_aligner, gen_head, gen_embed


@dataclass
class JanusVQ16DecoderOutput(BaseDecoderOutput):
    losses: Optional[Dict] = None


class JanusVQ16Decoder(BaseDecoderModelMixin, VQModel):
    config_class = JanusVQ16DecoderConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: JanusVQ16DecoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if config.projector_train_from_scratch:
            self.customized_gen_aligner, self.customized_gen_head, self.customized_gen_embed = build_projector(
                config, config.output_size
            )
        else:
            self.gen_aligner, self.gen_head, self.gen_embed = build_projector(config, config.n_embed)
            if config.add_projector and config.output_size is not None:
                self.encoder_projector = build_feature_projector(config.n_embed, config.output_size)
                self.decoder_projector = build_feature_projector(config.output_size, config.n_embed)
            else:
                if config.output_size is not None and config.output_size != config.n_embed:
                    raise ValueError("`output_size` should be same as `hidden_size`.")
                self.encoder_projector = nn.Identity()
                self.decoder_projector = nn.Identity()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        if self.config.projector_train_from_scratch:
            self.customized_gen_aligner.requires_grad_(True)
            self.customized_gen_head.requires_grad_(True)
            self.customized_gen_embed.requires_grad_(True)
        else:
            if self.config.output_size is not None and self.config.add_projector:
                self.encoder_projector.requires_grad_(True)
                self.decoder_projector.requires_grad_(True)
                if self.config.train_origin_projector:
                    self.gen_aligner.requires_grad_(True)
                    self.gen_head.requires_grad_(True)
                    self.gen_embed.requires_grad_(True)
            else:
                self.gen_aligner.requires_grad_(True)
                self.gen_head.requires_grad_(True)
                self.gen_embed.requires_grad_(True)

    def lm_encode(self, features: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = features.shape[0]
        _, _, info = self.encode(features)
        indices = info[-1]
        token_num = indices.shape[0] // bs
        assert token_num in [1, 576]  # 1 for fake input
        indices = rearrange(indices, "(b d) -> b d", b=bs, d=token_num)
        if self.config.projector_train_from_scratch:
            embeds = self.customized_gen_aligner(self.customized_gen_embed(indices))
        else:
            embeds = self.encoder_projector(self.gen_aligner(self.gen_embed(indices)))
        embeds = rearrange(embeds, "b d c -> (b d) c", b=bs, d=token_num)
        indices = rearrange(indices, "b d -> (b d)", b=bs, d=token_num)
        return embeds, indices

    def lm_head(self, hidden_states: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if self.config.projector_train_from_scratch:
            logits = self.customized_gen_head(hidden_states)
        else:
            projected = self.decoder_projector(hidden_states)
            logits = self.gen_head(projected)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            if get_parallel_state().sp_enabled:
                num_valid_tokens = (labels != -100).sum()
                loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
        return JanusVQ16DecoderOutput(
            loss=loss,
            logits=logits,
        )

    def embed_to_indice(
        self, hidden_states: torch.Tensor, temperature: float = 1.0, cfg: bool = False, cfg_weight: int = 5
    ):
        if self.config.projector_train_from_scratch:
            logits = self.customized_gen_head(hidden_states)
        else:
            logits = self.gen_head(self.decoder_projector(hidden_states))
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
        next_token = next_token.view(bs, -1)
        return logits, next_token

    def lm_embed(self, hidden_states: torch.Tensor, **kwargs):
        logits, indices = self.embed_to_indice(hidden_states, **kwargs)
        if self.config.projector_train_from_scratch:
            embeds = self.customized_gen_aligner(self.customized_gen_embed(indices))
        else:
            embeds = self.encoder_projector(self.gen_aligner(self.gen_embed(indices)))  # bs n dim
        return embeds, indices  # bs n dim, bs

    def lm_generate(self, indices: torch.Tensor, **kwargs):
        indices = indices.reshape(-1, 24, 24)
        bs = indices.shape[0]
        return self.decode_code(indices, shape=[bs, 8, 24, 24])

    def _get_lm_dummy_data(self) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn((1, 3, 384, 384), dtype=self.dtype, device=self.device)
        return {"features": pixel_values}
