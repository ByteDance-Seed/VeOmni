"""Configuration for the MiniMax-H3-Image VAE."""

from transformers import PretrainedConfig


class MiniMaxH3ImageVAEConfig(PretrainedConfig):
    model_type = "MiniMaxH3ImageVAE"

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 24,
        decoder_num_attention_heads: int = 16,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.5,
        decoder_norm_eps: float = 1e-5,
        latents_mean: tuple[float, ...] = (0.0,) * 24,
        latents_std: tuple[float, ...] = (1.0,) * 24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = tuple(block_out_channels)
        self.layers_per_block = layers_per_block
        self.spatial_downsample_factors = tuple(spatial_downsample_factors)
        self.temporal_downsample_factors = tuple(temporal_downsample_factors)
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.spatial_padding_mode = spatial_padding_mode
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_attention_head_dim = decoder_attention_head_dim
        self.decoder_num_register_tokens = decoder_num_register_tokens
        self.decoder_ffn_mult = decoder_ffn_mult
        self.decoder_rope_theta = decoder_rope_theta
        self.decoder_rope_dim_ratio = decoder_rope_dim_ratio
        self.decoder_norm_eps = decoder_norm_eps
        self.latents_mean = tuple(latents_mean)
        self.latents_std = tuple(latents_std)
