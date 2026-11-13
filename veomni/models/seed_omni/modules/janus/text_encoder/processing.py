"""Worker-side preprocessor for :class:`JanusTextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import JanusChatTemplate


class JanusTextEncoderPreprocessor(TextEncoderPreprocessor):
    """Janus chat template + tokenize worker.

    Only the chat-template binding differs from the shared text-encoder
    preprocessor: Janus has its own ``<boi>`` / ``<eoi>`` markers.
    """

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        del config_overrides
        return JanusChatTemplate(build_tokenizer(module_path))


__all__ = ["JanusTextEncoderPreprocessor"]
