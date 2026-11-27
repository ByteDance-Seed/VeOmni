"""Accelerated runtime for the qwen3omni/talker OmniModule."""

from .accelerated import (
    Qwen3OmniTalkerAccelerated,
    TrainingMixin,
    VeOmniMixin,
)


__all__ = [
    "Qwen3OmniTalkerAccelerated",
    "TrainingMixin",
    "VeOmniMixin",
]
