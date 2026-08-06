from ...base.text_encoder.modeling import TextEncoder
from .configuration import Qwen3TextEncoderConfig
from .modulemixin import VeOmniMixin


class Qwen3TextEncoder(VeOmniMixin, TextEncoder):
    config_class = Qwen3TextEncoderConfig

    def __init__(self, config: Qwen3TextEncoderConfig):
        super().__init__(config)
