"""Config for :class:`Qwen3VLTextEncoder`."""

from ...base.text_encoder.configuration import TextEncoderConfig


class Qwen3VLTextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for Qwen3-VL ChatML tokenization (text + image)."""

    model_type = "qwen3vl_text_encoder"
