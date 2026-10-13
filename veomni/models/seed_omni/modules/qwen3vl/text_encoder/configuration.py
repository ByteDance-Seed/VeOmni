"""Config for :class:`Qwen3VLTextEncoder`."""

from ...base.text_encoder.configuration import TextEncoderConfig


class Qwen3VLTextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for Qwen3-VL ChatML tokenization (text + image)."""

    model_type = "qwen3vl_text_encoder"

    # transformers v5 swaps an ``__init__``-less config subclass for a dataclass
    # one that skips the parents' ``__init__`` (and with it their defaults).
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
