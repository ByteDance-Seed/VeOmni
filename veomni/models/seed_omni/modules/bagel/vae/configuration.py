"""Configuration for BAGEL's latent VAE module."""

from transformers import PretrainedConfig


class BagelVAEConfig(PretrainedConfig):
    """BAGEL FLUX-style latent autoencoder config."""

    model_type = "bagel_vae"

    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        downsample: int = 8,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        z_channels: int = 16,
        scale_factor: float = 0.3611,
        shift_factor: float = 0.1159,
        max_image_size: int = 1024,
        min_image_size: int = 512,
        image_stride: int = 16,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        freeze: bool = True,
        **kwargs,
    ) -> None:
        self.resolution = resolution
        self.in_channels = in_channels
        self.downsample = downsample
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = [1, 2, 4, 4] if ch_mult is None else ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.image_stride = image_stride
        self.max_pixels = max_pixels
        self.image_mean = [0.5, 0.5, 0.5] if image_mean is None else image_mean
        self.image_std = [0.5, 0.5, 0.5] if image_std is None else image_std
        self.freeze = freeze
        super().__init__(**kwargs)


__all__ = ["BagelVAEConfig"]
