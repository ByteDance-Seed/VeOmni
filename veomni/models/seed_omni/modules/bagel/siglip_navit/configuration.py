"""Configuration for BAGEL's SigLIP NaViT module."""

from transformers import PretrainedConfig


class BagelSiglipNavitConfig(PretrainedConfig):
    model_type = "bagel_siglip_navit"

    def __init__(
        self,
        hidden_size: int = 1152,
        image_size: int = 980,
        intermediate_size: int = 4304,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 26,
        num_channels: int = 3,
        patch_size: int = 14,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        rope: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope = rope
        super().__init__(**kwargs)
