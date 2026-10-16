"""Accelerated runtime for the janus/text_encoder OmniModule (conversation + packed graphs)."""

from .accelerated import (
    JanusTextEncoderAccelerated,
)


__all__ = [
    "JanusTextEncoderAccelerated",
]
