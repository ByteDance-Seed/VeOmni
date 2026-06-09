"""BAGEL token embedding and LM head module."""

from ...base.text_encoder.modeling import TextEncoder
from .configuration import BagelTextEncoderConfig


class BagelTextEncoder(TextEncoder):
    config_class = BagelTextEncoderConfig
