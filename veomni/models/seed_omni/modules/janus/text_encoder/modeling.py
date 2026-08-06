from ...base.text_encoder.modeling import TextEncoder
from .configuration import JanusTextEncoderConfig
from .modulemixin import VeOmniMixin


class JanusTextEncoder(VeOmniMixin, TextEncoder):
    config_class = JanusTextEncoderConfig

    def __init__(self, config: JanusTextEncoderConfig):
        super().__init__(config)
