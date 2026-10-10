"""Accelerated runtime for the base/text_encoder OmniModule (conversation graph; the packed hooks live in the family subclasses)."""

from .accelerated import (
    TextEncoderAccelerated,
    TrainingMixin,
    VeOmniMixin,
    scatter_text_encoder_embeds,
)


__all__ = [
    "TextEncoderAccelerated",
    "TrainingMixin",
    "VeOmniMixin",
    "scatter_text_encoder_embeds",
]
