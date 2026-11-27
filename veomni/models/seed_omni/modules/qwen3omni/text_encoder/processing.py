"""Worker-side preprocessor for :class:`Qwen3OmniTextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import Qwen3OmniChatTemplate
from .configuration import Qwen3OmniTextEncoderConfig


class Qwen3OmniTextEncoderPreprocessor(TextEncoderPreprocessor):
    """Qwen3-Omni ChatML chat template + tokenize worker.

    Only the text pipeline runs here. Media rows stay on the carrier with the
    patches / mel features their own tower's preprocessor left on them, and the
    backbone packs the segments at forward time — see
    :func:`~veomni.models.seed_omni.modules.qwen3omni.llm.modeling.pack_qwen3omni_conversations_for_forward`.
    """

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        # Read from the checkpoint config so the CPU-side template matches the
        # module's own, with the YAML ``model_config:`` block layered on top —
        # a mismatch would template the prompt differently from how the round
        # was trained.
        config = Qwen3OmniTextEncoderConfig.from_pretrained(module_path, **(config_overrides or {}))
        return Qwen3OmniChatTemplate(
            build_tokenizer(module_path),
            add_speech_system_prompt=config.add_speech_system_prompt,
        )


__all__ = ["Qwen3OmniTextEncoderPreprocessor"]
