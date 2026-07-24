# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local Transformers 5 configuration for the HunyuanImage 3 training model.

Adapted from Tencent's ``configuration_hunyuan_image_3.py`` at revision
``6e9113a692a27a0751d82aba3b2015a876646c03``. The text-backbone fields are
normalized onto Transformers 5.9 ``HunYuanMoEV1Config`` while image-generation
and component-lifecycle fields remain explicit local extensions.
"""

from copy import deepcopy
from typing import Any, Mapping

from transformers.models.hunyuan_v1_moe.configuration_hunyuan_v1_moe import HunYuanMoEV1Config

from .component_policy import HunyuanImage3ComponentPolicy


DEFAULT_COMPONENT_POLICY = {
    "transformer": "trainable",
    "text_embedding": "trainable",
    "image_projector": "trainable",
    "timestep_modules": "trainable",
    "image_head": "trainable",
    "vae_encoder": "absent",
    "vae_decoder": "absent",
    "vision_model": "absent",
    "vision_aligner": "absent",
    "lm_head": "absent",
}

DEFAULT_IMAGE_ROPE_SCALING = {
    "factor": 1.0,
    "type": "custom",
}

DEFAULT_VAE_CONFIG = {
    "block_out_channels": [128, 256, 512, 1024, 1024],
    "downsample_match_channel": True,
    "ffactor_spatial": 16,
    "ffactor_temporal": 4,
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "out_channels": 3,
    "sample_size": 384,
    "sample_tsize": 96,
    "scaling_factor": 0.562679178327931,
    "shift_factor": None,
}


class HunyuanImage3Config(HunYuanMoEV1Config):
    model_type = "hunyuan_image_3_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        kwargs.pop("model_type", None)
        kwargs.pop("auto_map", None)

        attention_head_dim = kwargs.pop("attention_head_dim", kwargs.pop("head_dim", None))
        hidden_size = kwargs.pop("hidden_size", 4096)
        num_attention_heads = kwargs.pop("num_attention_heads", 32)
        if attention_head_dim is None:
            attention_head_dim = hidden_size // num_attention_heads

        intermediate_size = kwargs.pop("intermediate_size", 3072)
        moe_intermediate_size = kwargs.pop("moe_intermediate_size", None)
        if moe_intermediate_size is None:
            moe_intermediate_size = intermediate_size
        if isinstance(moe_intermediate_size, list):
            if not moe_intermediate_size or any(size != moe_intermediate_size[0] for size in moe_intermediate_size):
                raise ValueError(
                    "The initial HunyuanImage 3 runtime requires one MoE intermediate size for all layers."
                )
            routed_intermediate_size = moe_intermediate_size[0]
        else:
            routed_intermediate_size = moe_intermediate_size
        if routed_intermediate_size != intermediate_size:
            raise ValueError(
                "The initial HunyuanImage 3 runtime requires shared and routed MLP intermediate sizes to match."
            )

        num_hidden_layers = kwargs.pop("num_hidden_layers", 32)
        num_shared_expert = kwargs.pop("num_shared_expert", 1)
        shared_experts = (
            num_shared_expert if isinstance(num_shared_expert, list) else [num_shared_expert] * num_hidden_layers
        )
        if len(shared_experts) != num_hidden_layers or any(count != 1 for count in shared_experts):
            raise ValueError("Transformers 5.9 HunYuanMoEV1 supports exactly one shared MLP per layer.")

        # The official Base config stores per-layer lists for MoE fields, but the
        # Transformers 5.9 HunYuanMoEV1 runtime consumes scalars. Collapse the
        # homogeneous official lists here so the real checkpoint config builds.
        moe_topk = kwargs.pop("moe_topk", 8)
        if isinstance(moe_topk, list):
            if not moe_topk or any(k != moe_topk[0] for k in moe_topk):
                raise ValueError("The initial HunyuanImage 3 runtime requires one MoE top-k for all layers.")
            moe_topk = moe_topk[0]

        use_mixed_mlp_moe = kwargs.pop("use_mixed_mlp_moe", True)
        use_qk_norm = kwargs.pop("use_qk_norm", True)
        if not use_mixed_mlp_moe:
            raise ValueError("The initial HunyuanImage 3 runtime requires the official shared MLP path.")
        if not use_qk_norm:
            raise ValueError("The initial HunyuanImage 3 runtime requires per-head QK normalization.")

        attention_dropout = kwargs.pop("attention_dropout", 0.0)
        if attention_dropout != 0.0:
            raise ValueError("HunyuanImage 3 training requires attention_dropout=0.0.")

        rope_theta = kwargs.pop("rope_theta", 10000.0)
        image_rope_scaling = kwargs.pop(
            "image_rope_scaling",
            kwargs.pop("rope_scaling", DEFAULT_IMAGE_ROPE_SCALING),
        )
        rope_parameters = kwargs.pop(
            "rope_parameters",
            {"rope_type": "default", "rope_theta": rope_theta},
        )

        component_policy_values = kwargs.pop("component_policy", DEFAULT_COMPONENT_POLICY)
        self.component_policy = HunyuanImage3ComponentPolicy.from_dict(component_policy_values).as_dict()

        vae = kwargs.pop("vae", DEFAULT_VAE_CONFIG)
        if not isinstance(vae, Mapping):
            raise TypeError("vae must be a mapping.")
        self.vae = deepcopy(dict(vae))
        self._validate_vae_config()

        self.model_version = kwargs.pop("model_version", "HunyuanImage-3.0")
        self.attention_head_dim = attention_head_dim
        # Runtime modeling reads a scalar routed intermediate size (validated homogeneous above).
        self.moe_intermediate_size = routed_intermediate_size
        self.num_shared_expert = deepcopy(num_shared_expert)
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = kwargs.pop("use_rotary_pos_emb", True)
        self.norm_topk_prob = kwargs.pop("norm_topk_prob", True)
        self.img_proj_type = kwargs.pop("img_proj_type", "unet")
        self.patch_size = kwargs.pop("patch_size", 1)
        self.patch_embed_hidden_dim = kwargs.pop("patch_embed_hidden_dim", 1024)
        self.rope_type = kwargs.pop("rope_type", "2d")
        self.image_rope_scaling = deepcopy(image_rope_scaling)
        self.vae_downsample_factor = tuple(kwargs.pop("vae_downsample_factor", (16, 16)))
        self.vae_dtype = kwargs.pop("vae_dtype", "float32")
        self.vae_autocast_dtype = kwargs.pop("vae_autocast_dtype", "float16")

        self.eod_token_id = kwargs.pop("eod_token_id", 3)
        self.im_start_id = kwargs.pop("im_start_id", 128000)
        self.im_end_id = kwargs.pop("im_end_id", 128001)
        self.text_start_id = kwargs.pop("text_start_id", 6)
        self.text_end_id = kwargs.pop("text_end_id", 7)
        self.image_token_id = kwargs.pop("image_token_id", 128006)
        self.pad_id = kwargs.pop("pad_id", kwargs.get("pad_token_id", 128009))

        architectures = kwargs.pop("architectures", ["HunyuanImage3ForCausalMM"])
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        common_kwargs = {
            name: kwargs.pop(name)
            for name in (
                "transformers_version",
                "output_hidden_states",
                "return_dict",
                "chunk_size_feed_forward",
                "is_encoder_decoder",
                "id2label",
                "label2id",
                "problem_type",
            )
            if name in kwargs
        }

        super().__init__(
            architectures=architectures,
            dtype=dtype,
            vocab_size=kwargs.pop("vocab_size", 133120),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=kwargs.pop("num_key_value_heads", 8),
            hidden_act=kwargs.pop("hidden_act", "silu"),
            max_position_embeddings=kwargs.pop("max_position_embeddings", 22800),
            initializer_range=kwargs.pop("initializer_range", 0.02),
            rms_norm_eps=kwargs.pop("rms_norm_eps", 1e-5),
            use_cache=kwargs.pop("use_cache", False),
            pad_token_id=kwargs.pop("pad_token_id", 128009),
            bos_token_id=kwargs.pop("bos_token_id", 127958),
            eos_token_id=kwargs.pop("eos_token_id", 127957),
            eod_token_id=self.eod_token_id,
            sep_token_id=kwargs.pop("sep_token_id", self.im_start_id),
            pretraining_tp=kwargs.pop("pretraining_tp", 1),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            rope_parameters=rope_parameters,
            attention_bias=kwargs.pop("attention_bias", False),
            attention_dropout=attention_dropout,
            num_experts=kwargs.pop("num_experts", 64),
            moe_topk=moe_topk,
            head_dim=attention_head_dim,
            **common_kwargs,
        )

        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.hidden_size != self.num_attention_heads * self.attention_head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * attention_head_dim for HunyuanImage 3.")
        if self.attention_head_dim % 4:
            raise ValueError("Hunyuan Image 3 generalized 2D RoPE requires attention_head_dim divisible by four.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.img_proj_type != "unet":
            raise ValueError("The initial HunyuanImage 3 runtime supports only img_proj_type='unet'.")
        if self.patch_size not in (1, 2, 4, 8):
            raise ValueError("patch_size must be one of 1, 2, 4, or 8.")
        if not self.use_rotary_pos_emb or self.rope_type != "2d":
            raise ValueError("The Hunyuan Image 3 reference path requires generalized 2D RoPE.")
        if not isinstance(self.image_rope_scaling, Mapping) or self.image_rope_scaling.get("type") != "custom":
            raise ValueError("Hunyuan Image 3 image_rope_scaling must select the custom reference layout.")
        rope_factor = self.image_rope_scaling.get("factor", 1.0)
        if isinstance(rope_factor, bool) or not isinstance(rope_factor, (int, float)) or rope_factor <= 0:
            raise ValueError("image_rope_scaling.factor must be positive.")

    def _validate_vae_config(self) -> None:
        required = {
            "block_out_channels",
            "downsample_match_channel",
            "ffactor_spatial",
            "ffactor_temporal",
            "in_channels",
            "latent_channels",
            "layers_per_block",
            "sample_size",
            "sample_tsize",
            "scaling_factor",
        }
        missing = sorted(required.difference(self.vae))
        if missing:
            raise ValueError(f"HunyuanImage 3 VAE config is missing fields: {missing}.")
        if self.vae["latent_channels"] <= 0:
            raise ValueError("vae.latent_channels must be positive.")


__all__ = [
    "DEFAULT_COMPONENT_POLICY",
    "DEFAULT_IMAGE_ROPE_SCALING",
    "DEFAULT_VAE_CONFIG",
    "HunyuanImage3Config",
]
