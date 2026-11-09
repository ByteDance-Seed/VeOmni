"""Worker-side preprocessor for :class:`Cosmos3TextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ....utils.conversation import ConversationItem
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import Cosmos3ChatTemplate
from .configuration import Cosmos3TextEncoderConfig


class Cosmos3TextEncoderPreprocessor(TextEncoderPreprocessor):
    """Tokenize the prompt in the DataLoader worker, so ``dyn_bsz`` prices it exactly."""

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> Cosmos3ChatTemplate:
        config = Cosmos3TextEncoderConfig.from_pretrained(module_path, model_config=config_overrides)
        return Cosmos3ChatTemplate(
            build_tokenizer(module_path),
            generation_marker_ids=config.generation_marker_ids,
            max_sequence_length=config.max_sequence_length,
        )

    def preprocess_conversations(
        self,
        conversation_list: list[list[ConversationItem]],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        # ``seedomni_cached`` replays samples the offline-embedding run already tokenized.
        pending = [sample for sample in conversation_list or [] if not _is_tokenized(sample)]
        super().preprocess_conversations(pending, inference=inference, **kwargs)


def _is_tokenized(sample: list[ConversationItem]) -> bool:
    return any(item.type == "text" and not isinstance(item.value, str) for item in sample)


__all__ = ["Cosmos3TextEncoderPreprocessor"]
