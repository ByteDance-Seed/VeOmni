from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3VLTextEncoderConfig
from .modulemixin import Qwen3VLTextEncoderModuleMixin


class Qwen3VLTextEncoder(Qwen3VLTextEncoderModuleMixin, TextEncoder):
    config_class = Qwen3VLTextEncoderConfig

    def __init__(self, config: Qwen3VLTextEncoderConfig):
        super().__init__(config)
