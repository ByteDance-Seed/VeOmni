"""Accelerated runtime for the bagel/qwen2_mot OmniModule (conversation graph)."""

from .accelerated import (
    BagelQwen2MoTAccelerated,
    InferenceMixinAccelerated,
)


__all__ = [
    "BagelQwen2MoTAccelerated",
    "InferenceMixinAccelerated",
]
