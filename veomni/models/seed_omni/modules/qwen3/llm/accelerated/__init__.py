"""Accelerated runtime for the qwen3/llm OmniModule (conversation graph)."""

from .accelerated import (
    Qwen3LlmAccelerated,
    VeOmniMixin,
)


__all__ = [
    "Qwen3LlmAccelerated",
    "VeOmniMixin",
]
