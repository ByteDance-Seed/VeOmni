from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3VLTextEncoderConfig
from .modulemixin import VeOmniMixin


class Qwen3VLTextEncoder(VeOmniMixin, TextEncoder):
    config_class = Qwen3VLTextEncoderConfig

    def __init__(self, config: Qwen3VLTextEncoderConfig):
        super().__init__(config)
