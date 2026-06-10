"""BAGEL token embedding and LM head module."""

from ...base.text_encoder.modeling import TextEncoder
from .configuration import BagelTextEncoderConfig
from .modulemixin import BagelTextEncoderModuleMixin


class BagelTextEncoder(BagelTextEncoderModuleMixin, TextEncoder):
    config_class = BagelTextEncoderConfig
