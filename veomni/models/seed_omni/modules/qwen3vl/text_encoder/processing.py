"""Worker-side preprocessor for :class:`Qwen3VLTextEncoder` — HF ``AutoProcessor`` style.

:class:`Qwen3VLTextEncoderPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.processing.base.ModulePreprocessorBase`).
Its :meth:`from_pretrained` loads the tokenizer and builds the
:class:`~veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template.Qwen3VLChatTemplate`
straight off the checkpoint dir — no model instance involved.
"""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ....processing import ModulePreprocessorBase
from ....utils.conversation import ConversationItem
from .chat_template import Qwen3VLChatTemplate


class Qwen3VLTextEncoderPreprocessor(ModulePreprocessorBase):
    """Worker-side ``apply_chat_template`` → tokenize → merge for the Qwen3-VL text encoder.

    Holds only the (picklable) :class:`Qwen3VLChatTemplate` — never the model — so it
    runs in DataLoader workers and overlaps with GPU compute. Builds CPU tensors;
    the main process's thin ``encode_pre`` does the single ``.to(device)``.
    """

    def __init__(self, chat_template: Qwen3VLChatTemplate) -> None:
        self._chat_template = chat_template
        # bind_module_assets also copies _tokenizer onto the model — TextEncoderModuleMixin
        # (base/text_encoder/modulemixin.py) decodes generated text via self._tokenizer.
        self._tokenizer = chat_template.tokenizer

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> Qwen3VLTextEncoderPreprocessor:
        """Build straight from the checkpoint dir — no model instance needed."""
        del kwargs
        tokenizer = build_tokenizer(module_path)
        return cls(Qwen3VLChatTemplate(tokenizer))

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, add_generation_prompt=inference)
            sample.clear()
            sample.extend(parts)


__all__ = ["Qwen3VLTextEncoderPreprocessor"]
