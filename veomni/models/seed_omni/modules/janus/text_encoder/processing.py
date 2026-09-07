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
        if not self.packed_preprocess or inference:
            super().__call__(batch, inference=inference, **kwargs)
            return

        # Packed training feeds the tokenized parts straight to the packer rather
        # than writing them back onto ``conversation_list`` (what the base
        # ``preprocess_conversations`` does) and then walking that list again.
        #
        # The carrier is dropped once packed: no node in ``graph_train_packed.yaml``
        # reads ``conversation_list``, and keeping it would ship every per-item
        # pixel tensor to the main process a second time — ``pack_janus_conversations``
        # already *copied* those pixels into ``pixel_values_und`` / ``pixel_values_gen``
        # via ``torch.stack``, so the carrier's references only pin a duplicate
        # (~400 MB per micro-batch at mbs=448). This is why the module must be the
        # last preprocessor in ``config.modules`` order: SigLIP/VQVAE write the pixel
        # tensors onto the items this reads. ``offline_cache_step`` needs the carrier,
        # but it is a conversation-path feature — the packed graph never fills items.
        tc_kwargs = self._tokenize_conversation_kwargs(inference, **kwargs)
        conversation_list = batch.pop("conversation_list")
        batch.update(
            pack_janus_conversations(
                (self._chat_template.tokenize_conversation(sample, **tc_kwargs) for sample in conversation_list),
                pad_token_id=int(self._chat_template.pad_token_id),
                num_image_tokens=self._num_image_tokens,
            )
        )


__all__ = ["JanusTextEncoderPreprocessor"]
