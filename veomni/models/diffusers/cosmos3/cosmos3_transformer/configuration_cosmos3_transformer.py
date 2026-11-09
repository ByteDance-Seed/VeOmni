"""Configuration for the Cosmos3 mixture-of-towers DiT. Defaults are Cosmos3-Nano's."""

from __future__ import annotations

from typing import Optional, Sequence

from veomni.models.configuration_base import VeOmniPretrainedConfig


class Cosmos3MoTTransformerConfig(VeOmniPretrainedConfig):
    model_type = "Cosmos3MoTTransformer"
    condition_model_type = "Cosmos3ConditionModel"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 12288,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        use_und_k_norm_for_gen: bool = True,
        patch_latent_dim: int = 192,
        latent_patch_size: int = 2,
        timestep_scale: float = 0.001,
        rope_theta: float = 5_000_000.0,
        rope_scaling: Optional[dict] = None,
        mrope_section: Optional[Sequence[int]] = None,
        mrope_interleaved: Optional[bool] = None,
        base_fps: float = 24.0,
        enable_fps_modulation: bool = True,
        temporal_compression_factor: int = 4,
        unified_3d_mrope_temporal_modality_margin: int = 0,
        train_joint_reasoner: bool = False,
        loss_scale: float = 10.0,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

        if patch_latent_dim % (latent_patch_size**2):
            raise ValueError(
                f"patch_latent_dim={patch_latent_dim} must be a multiple of latent_patch_size**2={latent_patch_size**2}."
            )

        # The checkpoint keeps the mRoPE knobs under ``rope_scaling``; flat fields win when set.
        self.rope_scaling = dict(rope_scaling) if rope_scaling else None
        scaling = self.rope_scaling or {}
        self.mrope_section = tuple(int(size) for size in (mrope_section or scaling.get("mrope_section", (24, 20, 20))))
        self.mrope_interleaved = bool(
            scaling.get("mrope_interleaved", True) if mrope_interleaved is None else mrope_interleaved
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.use_und_k_norm_for_gen = use_und_k_norm_for_gen
        self.patch_latent_dim = patch_latent_dim
        self.latent_patch_size = latent_patch_size
        self.latent_channel = patch_latent_dim // latent_patch_size**2
        self.timestep_scale = timestep_scale
        self.rope_theta = rope_theta
        self.base_fps = float(base_fps)
        self.enable_fps_modulation = enable_fps_modulation
        self.temporal_compression_factor = temporal_compression_factor
        self.unified_3d_mrope_temporal_modality_margin = unified_3d_mrope_temporal_modality_margin
        self.train_joint_reasoner = train_joint_reasoner
        self.loss_scale = loss_scale

    def to_dict(self) -> dict:
        config_dict = super().to_dict()
        config_dict["_class_name"] = self.model_type
        return config_dict
