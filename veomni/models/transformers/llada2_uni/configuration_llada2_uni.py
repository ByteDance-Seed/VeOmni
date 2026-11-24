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
"""Configuration for LLaDA2.0-Uni image understanding: the LLaDA2 MoE mask predictor plus
the frozen GlmImage tokenizer that turns an image into VQ token ids.

The text fields keep the checkpoint's ``llada2_moe`` names. ``vision_config`` and
``vq_config`` are the checkpoint's ``image_tokenizer/config.json`` sub-configs; VQ code
``c`` is the vocabulary token ``image_token_offset + c``.
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.models.glm_image.configuration_glm_image import GlmImageVisionConfig, GlmImageVQVAEConfig

from veomni.models.configuration_base import VeOmniPretrainedConfig


@strict
class LLaDA2UniConfig(VeOmniPretrainedConfig, PreTrainedConfig):
    r"""LLaDA2.0-Uni config.

    use_qk_norm (`bool`): RMSNorm over the head dim of queries and keys.
    use_qkv_bias / use_bias (`bool`): bias on the fused ``query_key_value`` / the ``dense`` output projection.
    num_shared_experts (`int`): width of the always-on expert, in units of ``moe_intermediate_size``.
    n_group / topk_group (`int`): the router keeps the ``topk_group`` best of ``n_group`` expert groups.
    routed_scaling_factor (`float`): scale on the normalized routed-expert weights.
    first_k_dense_replace (`int`): the first layers use a dense MLP of ``intermediate_size``.
    """

    model_type = "llada2_uni"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"vision_config": GlmImageVisionConfig, "vq_config": GlmImageVQVAEConfig}

    vocab_size: int = 173568
    hidden_size: int = 2048
    intermediate_size: int = 5120
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    use_qkv_bias: bool = False
    use_qk_norm: bool = True
    use_bias: bool = False
    rms_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    max_position_embeddings: int = 8192
    rope_parameters: RopeParameters | dict | None = None
    use_cache: bool = False
    tie_word_embeddings: bool = False
    pad_token_id: int | None = 156892
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 156892
    num_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    moe_intermediate_size: int = 512
    first_k_dense_replace: int = 1
    mask_token_id: int = 156895
    image_token_offset: int = 157184
    vision_config: dict | PreTrainedConfig | None = None
    vq_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 600000.0, "partial_rotary_factor": 0.5}
        if self.vision_config is None:
            self.vision_config = GlmImageVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = GlmImageVisionConfig(**self.vision_config)
        if self.vq_config is None:
            self.vq_config = GlmImageVQVAEConfig()
        elif isinstance(self.vq_config, dict):
            self.vq_config = GlmImageVQVAEConfig(**self.vq_config)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.num_experts % self.n_group:
            raise ValueError(f"`num_experts` ({self.num_experts}) must split evenly into `n_group` ({self.n_group}).")
        if self.vision_config.spatial_merge_size != 1:
            raise ValueError("The VQ codec reads one vision token per patch: `spatial_merge_size` must be 1.")
        if self.vision_config.hidden_size != self.vq_config.latent_channels:
            raise ValueError(
                f"The VQ codec reads {self.vq_config.latent_channels}-channel features, but the vision tower "
                f"emits {self.vision_config.hidden_size}."
            )
        if self.image_token_offset + self.vq_config.num_embeddings > self.vocab_size:
            raise ValueError("The VQ codes do not fit in the vocabulary above `image_token_offset`.")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rope_parameters.get("partial_rotary_factor", 1.0))


__all__ = ["LLaDA2UniConfig"]
