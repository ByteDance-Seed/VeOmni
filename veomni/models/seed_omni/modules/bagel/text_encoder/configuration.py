"""Configuration for :class:`BagelTextEncoder`."""

from ...base.text_encoder.configuration import TextEncoderConfig


class BagelTextEncoderConfig(TextEncoderConfig):
    """BAGEL text encoder config."""

    model_type = "bagel_text_encoder"

    def __init__(self, *args, tie_word_embeddings: bool = True, lm_head_bias: bool = False, **kwargs):
        super().__init__(*args, tie_word_embeddings=tie_word_embeddings, lm_head_bias=lm_head_bias, **kwargs)
