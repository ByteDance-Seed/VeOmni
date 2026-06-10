"""Config for :class:`Qwen3TextEncoder`."""

from ...base.text_encoder.configuration import TextEncoderConfig


class Qwen3TextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for Qwen3 ChatML tokenization."""

    model_type = "qwen3_text_encoder"
