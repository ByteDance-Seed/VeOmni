from ..base import BaseDecoderConfigMixin


class BagelVisionDecoderConfig(BaseDecoderConfigMixin):
    model_type = "bagel_vision_decoder"

    def __init__(
        self,
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
        latent_patch_size=2,
        timestep_shift=1.0,
        max_latent_size=32,
        **kwargs,
    ):
        self.resolution = resolution
        self.in_channels = in_channels
        self.downsample = downsample
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = list(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.latent_patch_size = latent_patch_size
        self.timestep_shift = timestep_shift
        self.max_latent_size = max_latent_size

        super().__init__(**kwargs)
