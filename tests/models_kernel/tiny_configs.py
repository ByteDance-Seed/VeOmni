# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Canonical tiny configs shared by models_kernel registry and parity tests."""

from __future__ import annotations

from transformers import PretrainedConfig


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121
AUDIO_TOKEN_ID = 122


def tiny_deepseek_v3_config(architecture: str = "DeepseekV3ForCausalLM") -> PretrainedConfig:
    """Build a four-layer toy that retains the official dense-to-MoE schedule."""
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=16,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        architectures=[architecture],
        attn_implementation="eager",
        experts_implementation="eager",
    )


def tiny_deepseek_v4_config(architecture: str = "DeepseekV4ForCausalLM") -> PretrainedConfig:
    """Build a four-layer toy that retains the official DSV4 layer schedules."""
    from veomni.models_kernel.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        max_position_embeddings=64,
        o_groups=8,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=16,
        architectures=[architecture],
        attn_implementation="eager",
        experts_implementation="eager",
    )


def tiny_flux_config(architecture: str = "FluxModel") -> PretrainedConfig:
    from veomni.models_kernel.transformers.flux.config_flux import FluxConfig

    return FluxConfig(
        disable_guidance_embedder=False,
        input_dim=16,
        output_dim=16,
        num_blocks=1,
        num_single_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        joint_attention_dim=48,
        pooled_projection_dim=32,
        timestep_embedding_dim=32,
        axes_dims_rope=(4, 6, 6),
        architectures=[architecture],
        tie_word_embeddings=False,
    )


def tiny_glm_moe_dsa_config(architecture: str = "GlmMoeDsaForCausalLM") -> PretrainedConfig:
    """Build a four-layer toy that retains GLM's dense-to-MoE schedule."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        index_topk=4,
        index_head_dim=16,
        index_n_heads=2,
        architectures=[architecture],
        attn_implementation="eager",
        experts_implementation="eager",
    )


def tiny_gemma3_text_config(
    architecture: str = "Gemma3ForCausalLM",
    *,
    layer_types: list[str] | None = None,
    final_logit_softcapping: float | None = None,
    attn_logit_softcapping: float | None = None,
) -> PretrainedConfig:
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

    if layer_types is None:
        layer_types = ["sliding_attention", "sliding_attention", "full_attention"]
    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=16,
        sliding_window=8,
        layer_types=layer_types,
        final_logit_softcapping=final_logit_softcapping,
        attn_logit_softcapping=attn_logit_softcapping,
        use_bidirectional_attention=False,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
    )


def tiny_gpt_oss_config(architecture: str = "GptOssForCausalLM") -> PretrainedConfig:
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    return GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=True,
        attention_dropout=0.0,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.001,
        output_router_logits=False,
        sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
        _experts_implementation="eager",
    )


def tiny_llama_config(architecture: str = "LlamaForCausalLM", **overrides) -> PretrainedConfig:
    from transformers.models.llama.configuration_llama import LlamaConfig

    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "attention_bias": False,
        "mlp_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "architectures": [architecture],
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def tiny_movqgan_config(architecture: str = "MoVQGAN") -> PretrainedConfig:
    from veomni.models_kernel.transformers.movqgan.configuration_movqgan import MoVQGANConfig

    return MoVQGANConfig(
        embed_dim=4,
        n_embed=16,
        double_z=False,
        z_channels=4,
        resolution=8,
        in_channels=3,
        out_ch=3,
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        dropout=0.0,
        architectures=[architecture],
        tie_word_embeddings=False,
    )


def tiny_seed_oss_config(architecture: str = "SeedOssForCausalLM") -> PretrainedConfig:
    from transformers.models.seed_oss.configuration_seed_oss import SeedOssConfig

    return SeedOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=True,
        attention_out_bias=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
    )


def tiny_wan_config(architecture: str = "WanModel", **overrides) -> PretrainedConfig:
    from veomni.models_kernel.transformers.wan.config_wan import WanConfig

    kwargs = {
        "patch_size": [1, 2, 2],
        "dim": 32,
        "eps": 1e-6,
        "ffn_dim": 64,
        "freq_dim": 16,
        "in_dim": 4,
        "num_heads": 4,
        "num_layers": 2,
        "out_dim": 4,
        "text_dim": 16,
        "text_len": 8,
        "has_image_input": "false",
        "architectures": [architecture],
        "tie_word_embeddings": False,
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return WanConfig(**kwargs)


def tiny_wan_t2v_config(architecture: str = "WanTransformer3DModel", **overrides) -> PretrainedConfig:
    from veomni.models_kernel.diffusers.wan_t2v.wan_transformer.configuration_wan_transformer import (
        WanTransformer3DModelConfig,
    )

    kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 4,
        "attention_head_dim": 16,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 32,
        "freq_dim": 16,
        "ffn_dim": 64,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 64,
        "architectures": [architecture],
        "tie_word_embeddings": False,
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return WanTransformer3DModelConfig(**kwargs)


def tiny_wan_t2v_condition_config(
    architecture: str = "WanTransformer3DConditionModel", **overrides
) -> PretrainedConfig:
    from veomni.models_kernel.diffusers.wan_t2v.wan_condition.configuration_wan_condition import (
        WanTransformer3DConditionModelConfig,
    )

    kwargs = {
        "base_model_path": "",
        "architectures": [architecture],
        "tie_word_embeddings": False,
    }
    kwargs.update(overrides)
    return WanTransformer3DConditionModelConfig(**kwargs)


def tiny_qwen2_config(architecture: str = "Qwen2ForCausalLM", **overrides) -> PretrainedConfig:
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "architectures": [architecture],
        "attn_implementation": "eager",
        "use_sliding_window": False,
    }
    kwargs.update(overrides)
    return Qwen2Config(**kwargs)


def tiny_qwen2_vl_config(architecture: str = "Qwen2VLForConditionalGeneration") -> PretrainedConfig:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import (
        Qwen2VLConfig,
        Qwen2VLTextConfig,
        Qwen2VLVisionConfig,
    )

    text = Qwen2VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_section": [4, 2, 2], "rope_type": "default"},
        tie_word_embeddings=False,
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_sliding_window=False,
    )
    vision = Qwen2VLVisionConfig(
        depth=2,
        embed_dim=64,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        hidden_act="quick_gelu",
    )
    return Qwen2VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        architectures=[architecture],
    )


def tiny_qwen2_5_vl_config(architecture: str = "Qwen2_5_VLForConditionalGeneration") -> PretrainedConfig:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )

    text = Qwen2_5_VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_section": [4, 2, 2], "rope_type": "default"},
        tie_word_embeddings=False,
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_sliding_window=False,
    )
    vision = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=16,
        out_hidden_size=64,
        fullatt_block_indexes=[0],
        hidden_act="silu",
    )
    return Qwen2_5_VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        architectures=[architecture],
    )


def tiny_qwen2_5_omni_text_config(
    architecture: str = "Qwen2_5OmniThinkerTextModel",
) -> PretrainedConfig:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig

    return Qwen2_5OmniTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_section": [4, 2, 2], "rope_type": "default"},
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_sliding_window=False,
    )


def tiny_qwen2_5_omni_thinker_config(
    architecture: str = "Qwen2_5OmniThinkerForConditionalGeneration",
) -> PretrainedConfig:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniVisionEncoderConfig,
    )

    text = tiny_qwen2_5_omni_text_config()
    vision = Qwen2_5OmniVisionEncoderConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=16,
        out_hidden_size=64,
        fullatt_block_indexes=[0],
        hidden_act="silu",
    )
    audio = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        d_model=16,
        output_dim=64,
        n_window=4,
        max_source_positions=16,
    )
    return Qwen2_5OmniThinkerConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        audio_config=audio.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        architectures=[architecture],
    )


def tiny_qwen2_5_omni_config(
    architecture: str = "Qwen2_5OmniForConditionalGeneration",
) -> PretrainedConfig:
    from veomni.models_kernel.transformers.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig

    thinker = tiny_qwen2_5_omni_thinker_config()
    return Qwen2_5OmniConfig(
        thinker_config=thinker.to_dict(),
        enable_audio_output=False,
        architectures=[architecture],
    )


def tiny_qwen3_config(architecture: str = "Qwen3ForCausalLM", **overrides) -> PretrainedConfig:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "architectures": [architecture],
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def tiny_qwen3_5_text_config(
    architecture: str = "Qwen3_5ForCausalLM",
    *,
    layer_types: list[str] | None = None,
) -> PretrainedConfig:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    layer_types = ["linear_attention"] if layer_types is None else layer_types
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
    )


def tiny_qwen3_5_config(
    architecture: str = "Qwen3_5ForConditionalGeneration",
    *,
    layer_types: list[str] | None = None,
) -> PretrainedConfig:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig

    text = tiny_qwen3_5_text_config(layer_types=layer_types)
    vision = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        hidden_act="gelu_pytorch_tanh",
    )
    return Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        architectures=[architecture],
    )


def tiny_qwen3_5_moe_text_config(
    architecture: str = "Qwen3_5MoeForCausalLM",
    *,
    layer_types: list[str] | None = None,
) -> PretrainedConfig:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    layer_types = ["linear_attention"] if layer_types is None else layer_types
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        experts_implementation="eager",
    )


def tiny_qwen3_5_moe_config(
    architecture: str = "Qwen3_5MoeForConditionalGeneration",
    *,
    layer_types: list[str] | None = None,
) -> PretrainedConfig:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeVisionConfig

    text = tiny_qwen3_5_moe_text_config(layer_types=layer_types)
    vision = Qwen3_5MoeVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        hidden_act="gelu_pytorch_tanh",
    )
    config = Qwen3_5MoeConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        architectures=[architecture],
    )
    config._experts_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    return config


def tiny_qwen3_moe_config(architecture: str = "Qwen3MoeForCausalLM", **overrides) -> PretrainedConfig:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "architectures": [architecture],
        "attn_implementation": "eager",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "experts_implementation": "eager",
    }
    kwargs.update(overrides)
    return Qwen3MoeConfig(**kwargs)


def tiny_qwen3_omni_moe_text_config(
    architecture: str = "Qwen3OmniMoeThinkerTextModel",
) -> PretrainedConfig:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

    config = Qwen3OmniMoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        architectures=[architecture],
        attn_implementation="eager",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coef=0.001,
    )
    config._experts_implementation = "eager"
    return config


def tiny_qwen3_omni_moe_thinker_config(
    architecture: str = "Qwen3OmniMoeThinkerForConditionalGeneration",
) -> PretrainedConfig:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
        Qwen3OmniMoeThinkerConfig,
        Qwen3OmniMoeVisionEncoderConfig,
    )

    text = tiny_qwen3_omni_moe_text_config()
    vision = Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh",
    )
    audio = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        d_model=16,
        output_dim=64,
        downsample_hidden_size=16,
        n_window=4,
        max_source_positions=16,
    )
    return Qwen3OmniMoeThinkerConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        audio_config=audio.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        architectures=[architecture],
    )


def tiny_qwen3_omni_moe_config(
    architecture: str = "Qwen3OmniMoeForConditionalGeneration",
) -> PretrainedConfig:
    from veomni.models_kernel.transformers.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeConfig,
    )

    thinker = tiny_qwen3_omni_moe_thinker_config()
    return Qwen3OmniMoeConfig(
        thinker_config=thinker.to_dict(),
        enable_audio_output=False,
        architectures=[architecture],
    )


def tiny_qwen3_vl_config(architecture: str = "Qwen3VLForConditionalGeneration") -> PretrainedConfig:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    text = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_interleaved": True, "mrope_section": [8, 4, 4], "rope_type": "default"},
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh",
    )
    return Qwen3VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        architectures=[architecture],
    )


def tiny_qwen3_vl_moe_config(
    architecture: str = "Qwen3VLMoeForConditionalGeneration",
) -> PretrainedConfig:
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
    )

    text = Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_interleaved": True, "mrope_section": [8, 4, 4], "rope_type": "default"},
        tie_word_embeddings=False,
        architectures=[architecture],
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        _experts_implementation="eager",
    )
    vision = Qwen3VLMoeVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh",
    )
    return Qwen3VLMoeConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        architectures=[architecture],
    )
