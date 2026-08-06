from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3TextEncoderConfig


class Qwen3TextEncoder(TextEncoder):
    config_class = Qwen3TextEncoderConfig

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__(config)
