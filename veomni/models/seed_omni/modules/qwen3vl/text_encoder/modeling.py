from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3VLTextEncoderConfig


class Qwen3VLTextEncoder(TextEncoder):
    config_class = Qwen3VLTextEncoderConfig

    def __init__(self, config: Qwen3VLTextEncoderConfig):
        super().__init__(config)
