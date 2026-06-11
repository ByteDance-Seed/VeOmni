"""Configuration for BAGEL's SigLIP NaViT module."""

from transformers import PretrainedConfig


class BagelSiglipNavitConfig(PretrainedConfig):
    model_type = "bagel_siglip_navit"

    def __init__(
        self,
        hidden_size: int = 1152,
        output_size: int = 3584,
        image_size: int = 980,
        min_image_size: int = 378,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        intermediate_size: int = 4304,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 26,
        num_channels: int = 3,
        patch_size: int = 14,
        hidden_act: str = "gelu_pytorch_tanh",
        connector_act: str = "gelu_pytorch_tanh",
        vit_max_num_patch_per_side: int = 70,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        rope: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.image_size = image_size
        self.min_image_size = min_image_size
        self.max_pixels = max_pixels
        self.image_mean = [0.5, 0.5, 0.5] if image_mean is None else image_mean
        self.image_std = [0.5, 0.5, 0.5] if image_std is None else image_std
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.connector_act = connector_act
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope = rope
        super().__init__(**kwargs)
