"""Accelerated runtime for the qwen3vl/text_encoder OmniModule (conversation + packed graphs)."""

from .accelerated import (
    Qwen3VLTextEncoderAccelerated,
)


__all__ = [
    "Qwen3VLTextEncoderAccelerated",
]
