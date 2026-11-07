"""Worker-side preprocessor for :class:`Qwen3TextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import Qwen3ChatTemplate, Qwen3VLChatTemplate
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
        """Pick the image-aware template when the module config says so.

        ``config_overrides`` mirrors the module's YAML ``model_config:`` block, so
        a worker built independently of the live model still selects the template
        that model was configured with.
        """
        config = Qwen3TextEncoderConfig.from_pretrained(module_path, **(config_overrides or {}))
        tokenizer = build_tokenizer(module_path)
        return Qwen3VLChatTemplate(tokenizer) if config.enable_image else Qwen3ChatTemplate(tokenizer)


__all__ = ["Qwen3TextEncoderPreprocessor"]
