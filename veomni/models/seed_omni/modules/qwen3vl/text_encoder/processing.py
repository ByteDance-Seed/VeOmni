"""Worker-side preprocessor for :class:`Qwen3VLTextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...base.text_encoder.chat_template import TextEncoderChatTemplate
from ...base.text_encoder.processing import TextEncoderPreprocessor
from .chat_template import Qwen3VLChatTemplate


class Qwen3VLTextEncoderPreprocessor(TextEncoderPreprocessor):
    """Qwen3-VL ChatML chat template + tokenize worker."""

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        del config_overrides
        return Qwen3VLChatTemplate(build_tokenizer(module_path))


__all__ = ["Qwen3VLTextEncoderPreprocessor"]
