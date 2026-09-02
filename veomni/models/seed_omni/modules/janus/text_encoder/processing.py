"""Worker-side preprocessor for :class:`JanusTextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...base.text_encoder.chat_template import TextEncoderChatTemplate
from ...base.text_encoder.processing import TextEncoderPreprocessor
from ..packing import JANUS_NUM_IMAGE_TOKENS, pack_janus_conversations
from .chat_template import JanusChatTemplate


class JanusTextEncoderPreprocessor(TextEncoderPreprocessor):
    """Janus chat template + tokenize worker.

    ``packed_preprocess=True`` (YAML ``processor_config``) tokenizes then
    writes packed ids / masks / stacked pixels onto the batch dict. Inference
    stays on the conversation-list path.
    """

    def __init__(
        self,
        chat_template: TextEncoderChatTemplate,
        *,
        packed_preprocess: bool = False,
        num_image_tokens: int = JANUS_NUM_IMAGE_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__(chat_template)
        del kwargs
        self.packed_preprocess = bool(packed_preprocess)
        self._num_image_tokens = int(num_image_tokens)

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        del config_overrides
        return JanusChatTemplate(build_tokenizer(module_path))

    def __call__(
        self,
        batch: dict[str, Any],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__call__(batch, inference=inference, **kwargs)
        if not self.packed_preprocess or inference:
            return
        packed = pack_janus_conversations(
            batch["conversation_list"],
            pad_token_id=int(self._chat_template.pad_token_id),
            num_image_tokens=self._num_image_tokens,
        )
        batch.update(packed)


__all__ = ["JanusTextEncoderPreprocessor"]
