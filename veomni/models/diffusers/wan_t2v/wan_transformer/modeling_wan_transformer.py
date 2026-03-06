from __future__ import annotations

import copy

from diffusers import WanTransformer3DModel as _WanTransformer3DModel
from transformers import PreTrainedModel

from .....utils import logging
from .configuration_wan_transformer import WanTransformer3DModelConfig


logger = logging.get_logger(__name__)


class WanTransformer3DModel(PreTrainedModel, _WanTransformer3DModel):
    config_class = WanTransformer3DModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: WanTransformer3DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        _WanTransformer3DModel.__init__(self, **config.to_diffuser_dict(), **kwargs)
        self.config: WanTransformer3DModelConfig = config
        self.config.tie_word_embeddings = False
        # inner_dim = config.num_attention_heads * config.attention_head_dim
        # out_channels = config.out_channels or config.in_channels

        # # 1. Patch & position embedding
        # self.rope = WanRotaryPosEmbed(config.attention_head_dim, config.patch_size, config.rope_max_seq_len)
        # self.patch_embedding = nn.Conv3d(config.in_channels, inner_dim, kernel_size=config.patch_size, stride=config.patch_size)

        # # 2. Condition embeddings
        # # image_embedding_dim=1280 for I2V model
        # self.condition_embedder = WanTimeTextImageEmbedding(
        #     dim=inner_dim,
        #     time_freq_dim=config.freq_dim,
        #     time_proj_dim=inner_dim * 6,
        #     text_embed_dim=config.text_dim,
        #     image_embed_dim=config.image_dim,
        #     pos_embed_seq_len=config.pos_embed_seq_len,
        # )

        # # 3. Transformer blocks
        # self.blocks = nn.ModuleList(
        #     [
        #         WanTransformerBlock(
        #             inner_dim, config.ffn_dim, config.num_attention_heads, config.qk_norm, config.cross_attn_norm, config.eps, config.added_kv_proj_dim
        #         )
        #         for _ in range(config.num_layers)
        #     ]
        # )

        # # 4. Output norm & projection
        # self.norm_out = FP32LayerNorm(inner_dim, config.eps, elementwise_affine=False)
        # self.proj_out = nn.Linear(inner_dim, config.out_channels * math.prod(config.patch_size))
        # self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # self.gradient_checkpointing = False

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    def forward(self, *args, **kwargs):
        return _WanTransformer3DModel.forward(self, *args, **kwargs)

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _WanTransformer3DModel.save_pretrained(self, path, **kwargs)
        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return _WanTransformer3DModel.from_pretrained(path, **kwargs)
