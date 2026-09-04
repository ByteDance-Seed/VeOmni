from transformers import PretrainedConfig


class MoVQGANConfig(PretrainedConfig):
    model_type = "movqgan"

    def __init__(
        self,
        embed_dim=4,
        n_embed=16384,
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=256,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(32,),
        dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        # base config
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        # ddconfig
        self.double_z = double_z
        self.z_channels = z_channels
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        # init config
        self.initializer_range = initializer_range
        super().__init__(**kwargs)
