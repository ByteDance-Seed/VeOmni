"""Config for :class:`Cosmos3Mot`. Defaults are Cosmos3-Nano's."""

from typing import Optional, Sequence

from ...module_configuration_base import OmniModuleConfig


class Cosmos3MotConfig(OmniModuleConfig):
    """The released ``transformer/config.json`` fields the decoder blocks and the mRoPE read.

    ``train_joint_reasoner`` releases the reasoner tower; by default only the
    generation tower trains, as in ``vision_sft_nano``.
    """

    model_type = "cosmos3_mot"

    def __init__(
        self,
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
        rope_theta: float = 5_000_000.0,
        mrope_section: Optional[Sequence[int]] = None,
        mrope_interleaved: bool = True,
        base_fps: float = 24.0,
        enable_fps_modulation: bool = True,
        temporal_compression_factor: int = 4,
        unified_3d_mrope_temporal_modality_margin: int = 0,
        latent_patch_size: int = 2,
        train_joint_reasoner: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.rope_theta = rope_theta
        self.mrope_section = [int(size) for size in (mrope_section or (24, 20, 20))]
        self.mrope_interleaved = mrope_interleaved
        self.base_fps = float(base_fps)
        self.enable_fps_modulation = enable_fps_modulation
        self.temporal_compression_factor = temporal_compression_factor
        self.unified_3d_mrope_temporal_modality_margin = unified_3d_mrope_temporal_modality_margin
        # The time axis is never patched; the MoT reads the latent grid with it.
        self.latent_patch_size = latent_patch_size
        self.train_joint_reasoner = train_joint_reasoner
