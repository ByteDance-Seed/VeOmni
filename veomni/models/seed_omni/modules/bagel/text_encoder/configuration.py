"""Configuration for BAGEL's token embedding and LM head module."""

from ...base.text_encoder.configuration import TextEncoderConfig


class BagelTextEncoderConfig(TextEncoderConfig):
    """BAGEL text vocab module.

    The official checkpoint has independent ``embed_tokens`` and ``lm_head``
    tensors, so ``tie_word_embeddings`` defaults to ``False``.
    """

    model_type = "bagel_text_encoder"

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        tie_word_embeddings: bool = False,
        lm_head_bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            lm_head_bias=lm_head_bias,
            **kwargs,
        )
