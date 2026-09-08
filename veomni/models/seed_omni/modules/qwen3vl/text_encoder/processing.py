"""Worker-side preprocessor for :class:`Qwen3VLTextEncoder`."""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...base.text_encoder.chat_template import TextEncoderChatTemplate
from ...base.text_encoder.processing import TextEncoderPreprocessor
from ..packing import pack_qwen3vl_conversations
from .chat_template import Qwen3VLChatTemplate


class Qwen3VLTextEncoderPreprocessor(TextEncoderPreprocessor):
    """Qwen3-VL ChatML chat template + tokenize worker.

    ``packed_preprocess=True`` (YAML ``processor_config``) tokenizes then
    writes packed ids / M-RoPE / concat patches onto the batch dict.
    Inference stays on the conversation-list path.
    """

    def __init__(
        self,
        chat_template: TextEncoderChatTemplate,
        *,
        packed_preprocess: bool = False,
        spatial_merge_size: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(chat_template)
        del kwargs
        self.packed_preprocess = bool(packed_preprocess)
        self._spatial_merge_size = int(spatial_merge_size)

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        del config_overrides
        return Qwen3VLChatTemplate(build_tokenizer(module_path))

    def __call__(
        self,
        batch: dict[str, Any],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        if not self.packed_preprocess or inference:
            super().__call__(batch, inference=inference, **kwargs)
            return

        # Packed training feeds tokenized parts straight to the packer. The
        # carrier is dropped once packed: no node in ``graph_train_packed.yaml``
        # reads ``conversation_list``, and keeping it would ship every per-item
        # patch tensor to the main process a second time.
        tc_kwargs = self._tokenize_conversation_kwargs(inference, **kwargs)
        conversation_list = batch.pop("conversation_list")
        tokenizer = self._chat_template.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self._chat_template.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        batch.update(
            pack_qwen3vl_conversations(
                (self._chat_template.tokenize_conversation(sample, **tc_kwargs) for sample in conversation_list),
                pad_token_id=int(pad_token_id),
                spatial_merge_size=self._spatial_merge_size,
            )
        )


__all__ = ["Qwen3VLTextEncoderPreprocessor"]
