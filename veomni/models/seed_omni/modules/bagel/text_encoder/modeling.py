"""BAGEL token embedding and LM head module."""

from ...base.text_encoder.modeling import TextEncoder
from .configuration import BagelTextEncoderConfig
from .modulemixin import VeOmniMixin


class BagelTextEncoder(VeOmniMixin, TextEncoder):
    config_class = BagelTextEncoderConfig


__all__ = ["BagelTextEncoder", "BagelTextEncoderConfig"]
