"""Accelerated runtime for the qwen3omni/codec OmniModule."""

from .accelerated import (
    Qwen3OmniCodecAccelerated,
    TrainingMixin,
    VeOmniMixin,
)


__all__ = [
    "Qwen3OmniCodecAccelerated",
    "TrainingMixin",
    "VeOmniMixin",
]
