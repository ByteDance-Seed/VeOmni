"""Worker-side preprocessor for :class:`JanusTextEncoder` — HF ``AutoProcessor`` style.

:class:`JanusTextEncoderPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.mixins.module_processor_mixin.Preprocessor`).
Its :meth:`from_pretrained` loads the tokenizer and builds the
:class:`~veomni.models.seed_omni.modules.janus.text_encoder.chat_template.JanusChatTemplate`
straight off the checkpoint dir — no model instance involved. Text encoders never
need a dummy branch (only image-modality preprocessors do).
"""

from typing import Any

from .....auto import build_tokenizer
from ....mixins.module_processor_mixin import Preprocessor
from ....utils.conversation import ConversationItem
from .chat_template import JanusChatTemplate


class JanusTextEncoderPreprocessor(Preprocessor):
    """Worker-side ``apply_chat_template`` → tokenize → merge for the Janus text encoder.

    Holds only the (picklable) tokenizer + :class:`JanusChatTemplate` — never the
    model — so it runs in DataLoader workers and overlaps with GPU compute.
    Builds CPU tensors; the main process's thin ``encode_pre`` does the single
    ``.to(device)``.
    """

    def __init__(self, chat_template: JanusChatTemplate) -> None:
        self._chat_template = chat_template
        # bind_preprocessor also copies _tokenizer onto the model — TextEncoderModuleMixin
        # (base/text_encoder/modulemixin.py) decodes generated text via self._tokenizer.
        self._tokenizer = chat_template.tokenizer

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> "JanusTextEncoderPreprocessor":
        """Build straight from the checkpoint dir — no model instance needed."""
        del kwargs
        tokenizer = build_tokenizer(module_path)
        return cls(JanusChatTemplate(tokenizer))

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, add_generation_prompt=inference)
            sample.clear()
            sample.extend(parts)


__all__ = ["JanusTextEncoderPreprocessor"]
