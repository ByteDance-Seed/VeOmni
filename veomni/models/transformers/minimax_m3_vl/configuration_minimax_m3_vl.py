# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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
from copy import deepcopy
from typing import Optional

from transformers import PretrainedConfig


def _drop_nested_model_type(config_dict):
    if config_dict is None:
        return None
    if isinstance(config_dict, PretrainedConfig):
        config_dict = config_dict.to_dict()
    else:
        config_dict = deepcopy(config_dict)
    config_dict.pop("model_type", None)
    return config_dict


def _default_rope_parameters(rope_parameters, rope_theta):
    if rope_parameters is None:
        return {"rope_theta": rope_theta, "rope_type": "default"}

    rope_parameters = deepcopy(rope_parameters)
    rope_parameters.setdefault("rope_theta", rope_theta)
    rope_parameters.setdefault("rope_type", "default")
    return rope_parameters


class MiniMaxM3VLTextConfig(PretrainedConfig):
    model_type = "minimax_m3_vl_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_experts": "num_local_experts",
    }
    base_config_key = "text_config"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.gate_up_proj_scale_inv": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj_scale_inv": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=6144,
        intermediate_size=3072,
        dense_intermediate_size=12288,
        shared_intermediate_size=3072,
        num_hidden_layers=60,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=524288,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=200034,
        eos_token_id=200020,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        num_experts_per_tok=4,
        num_local_experts=128,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        routed_scaling_factor=2.0,
        rotary_dim=64,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        rope_parameters=None,
        rope_scaling=None,
        rope_theta=5000000.0,
        sparse_attention_config=None,
        moe_layer_freq=None,
        mlp_layer_types=None,
        layer_types=None,
        index_n_heads=4,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
        **kwargs,
    ):
        num_experts = kwargs.pop("num_experts", None)
        if num_experts is not None:
            num_local_experts = num_experts
        sparse_attention_config = deepcopy(sparse_attention_config) if sparse_attention_config is not None else {}
        index_n_heads = sparse_attention_config.get("sparse_num_index_heads", index_n_heads)
        index_head_dim = sparse_attention_config.get("sparse_index_dim", index_head_dim)
        index_block_size = sparse_attention_config.get("sparse_block_size", index_block_size)
        index_topk_blocks = sparse_attention_config.get("sparse_topk_blocks", index_topk_blocks)
        index_local_blocks = sparse_attention_config.get("sparse_local_block", index_local_blocks)

        if layer_types is None and "sparse_attention_freq" in sparse_attention_config:
            layer_types = [
                "minimax_m3_sparse" if enabled else "full_attention"
                for enabled in sparse_attention_config["sparse_attention_freq"]
            ]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers

        if mlp_layer_types is None and moe_layer_freq is not None:
            mlp_layer_types = ["sparse" if enabled else "dense" for enabled in moe_layer_freq]
        if mlp_layer_types is None:
            mlp_layer_types = ["sparse"] * num_hidden_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = "silu" if hidden_act == "swigluoai" else hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.routed_scaling_factor = routed_scaling_factor
        self.rotary_dim = rotary_dim
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.rope_scaling = rope_scaling
        self.rope_parameters = _default_rope_parameters(rope_parameters, rope_theta)
        self.rope_theta = rope_theta
        self.sparse_attention_config = sparse_attention_config
        self.moe_layer_freq = moe_layer_freq
        self.mlp_layer_types = mlp_layer_types
        self.layer_types = layer_types
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_block_size = index_block_size
        self.index_topk_blocks = index_topk_blocks
        self.index_local_blocks = index_local_blocks


class MiniMaxM3VLVisionConfig(PretrainedConfig):
    model_type = "minimax_m3_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        image_size=2016,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        rope_parameters=None,
        rope_theta=10000.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope_parameters = _default_rope_parameters(rope_parameters, rope_theta)
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range


class MiniMaxM3VLConfig(PretrainedConfig):
    model_type = "minimax_m3_vl"
    sub_configs = {
        "text_config": MiniMaxM3VLTextConfig,
        "vision_config": MiniMaxM3VLVisionConfig,
    }
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=200025,
        video_token_index=200026,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        projector_hidden_size=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if image_token_id is not None:
            image_token_index = image_token_id
        if video_token_id is not None:
            video_token_index = video_token_id

        if isinstance(vision_config, MiniMaxM3VLVisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = MiniMaxM3VLVisionConfig(**(_drop_nested_model_type(vision_config) or {}))

        if isinstance(text_config, MiniMaxM3VLTextConfig):
            self.text_config = text_config
        else:
            self.text_config = MiniMaxM3VLTextConfig(**(_drop_nested_model_type(text_config) or {}))

        if projector_hidden_size is None:
            projector_hidden_size = self.text_config.hidden_size

        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.image_token_id = image_token_index
        self.video_token_id = video_token_index
        self.projector_hidden_size = projector_hidden_size
        self.merged_hidden_size = self.text_config.hidden_size * (self.vision_config.spatial_merge_size**2)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["MiniMaxM3VLConfig", "MiniMaxM3VLTextConfig", "MiniMaxM3VLVisionConfig"]
