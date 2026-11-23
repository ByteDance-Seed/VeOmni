"""Accelerated runtime for the qwen3omni/text_encoder OmniModule (conversation graph)."""

from .accelerated import (
    Qwen3OmniTextEncoderAccelerated,
    TrainingMixin,
    VeOmniMixin,
)


__all__ = [
    "Qwen3OmniTextEncoderAccelerated",
    "TrainingMixin",
    "VeOmniMixin",
]
