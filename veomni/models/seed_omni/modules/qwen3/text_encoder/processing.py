"""Worker-side preprocessor for :class:`Qwen3TextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...base.text_encoder.chat_template import TextEncoderChatTemplate
from ...base.text_encoder.processing import TextEncoderPreprocessor
from ...qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from .chat_template import Qwen3ChatTemplate
from .configuration import Qwen3TextEncoderConfig


class Qwen3TextEncoderPreprocessor(TextEncoderPreprocessor):
    """Qwen3 or Qwen3-VL (image mode) chat template + tokenize worker."""

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        config = Qwen3TextEncoderConfig.from_pretrained(module_path, **(config_overrides or {}))
        tokenizer = build_tokenizer(module_path)
        return Qwen3VLChatTemplate(tokenizer) if config.enable_image else Qwen3ChatTemplate(tokenizer)


__all__ = ["Qwen3TextEncoderPreprocessor"]
