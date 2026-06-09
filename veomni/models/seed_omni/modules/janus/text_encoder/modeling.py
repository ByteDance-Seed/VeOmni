from ...base.text_encoder.modeling import TextEncoder
from .configuration import JanusTextEncoderConfig
from .modulemixin import JanusTextEncoderModuleMixin


class JanusTextEncoder(JanusTextEncoderModuleMixin, TextEncoder):
    config_class = JanusTextEncoderConfig

    def __init__(self, config: JanusTextEncoderConfig):
        super().__init__(config)
