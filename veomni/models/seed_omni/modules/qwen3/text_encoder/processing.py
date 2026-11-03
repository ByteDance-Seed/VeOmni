"""Worker-side preprocessor for :class:`Qwen3TextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import Qwen3ChatTemplate


class Qwen3TextEncoderPreprocessor(TextEncoderPreprocessor):
    """Qwen3 ChatML chat template + tokenize worker."""

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        del config_overrides
        return Qwen3ChatTemplate(build_tokenizer(module_path))


__all__ = ["Qwen3TextEncoderPreprocessor"]
