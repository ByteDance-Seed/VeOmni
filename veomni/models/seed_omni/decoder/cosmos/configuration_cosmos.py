from transformers import PretrainedConfig

from .....utils import logging


logger = logging.get_logger(__name__)


class CosmosConfig(PretrainedConfig):
    model_type = "cosmos"

    def __init__(
        self,
        attn_resolutions=[32],  # The attention resolution for res blocks.
        channels=128,  # The base number of channels.
        channels_mult=[2, 4, 4],  # The channel multipler for each resolution.
        dropout=0.0,
        in_channels=3,
        spatial_compression=16,  # The spatial compression ratio.
        num_res_blocks=2,  # The number of layers in each res block.
        out_channels=3,
        resolution=1024,
        patch_size=4,
        patch_method="haar",
        z_channels=256,  # The encoder output channels just before sampling.
        z_factor=1,  # A factor over the z_channels, to get the total channels the encoder should output. for discrete tokenization, often we directly use the vector, so z_factor=1.
        quantizer="FSQ",  # The quantizer of choice, VQ, LFQ, FSQ, or ResFSQ.
        embedding_dim=6,  # The embedding dimension post-quantization, which is also the input channels of the decoder.Which is also the output
        levels=[8, 8, 8, 5, 5, 5],  # The number of levels to use for fine-scalar quantization.
        num_quantizers=4,  # The number of quantizers to use for residual fine-scalar quantization.
        name="DI",
        encoder="Default",  # Specify type of encoder ["Default", "LiteVAE"]
        decoder="Default",  # Specify type of decoder ["Default"]
        **kwargs,
    ):
        self.attn_resolutions = attn_resolutions
        self.channels = channels
        self.channels_mult = channels_mult
        self.dropout = dropout
        self.in_channels = in_channels
        self.spatial_compression = spatial_compression
        self.num_res_blocks = num_res_blocks
        self.out_channels = out_channels
        self.resolution = resolution
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.z_channels = z_channels
        self.z_factor = z_factor
        self.quantizer = quantizer
        self.embedding_dim = embedding_dim
        self.levels = levels
        self.num_quantizers = num_quantizers
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(**kwargs)
