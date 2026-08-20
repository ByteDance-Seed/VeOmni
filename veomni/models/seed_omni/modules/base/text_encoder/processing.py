"""Shared CPU worker for every SeedOmni text encoder module.

Each concrete text encoder keeps a thin ``processing.py`` that subclasses
:class:`TextEncoderPreprocessor` and implements :meth:`build_chat_template`
with its module-local :class:`~veomni.models.seed_omni.modules.base.text_encoder.chat_template.TextEncoderChatTemplate`
subclass.  Tokenize / merge / pack live on the base chat template; this worker
only runs ``tokenize_conversation`` inside DataLoader workers (training) or
before the generation FSM (inference).
"""

from __future__ import annotations

from typing import Any

from ....processing import ModulePreprocessorBase
from ....utils.conversation import ConversationItem
from .chat_template import TextEncoderChatTemplate


class TextEncoderPreprocessor(ModulePreprocessorBase):
    """Worker-side ``apply_chat_template`` → tokenize → merge for text encoders.

    Holds only the (picklable) chat template — never the model — so it runs in
    DataLoader workers and overlaps with GPU compute.  Builds CPU tensors; the
    main process's thin ``encode_pre`` does the single ``.to(device)``.

    :meth:`bind_module_assets` also copies ``_tokenizer`` onto the model —
    :class:`~veomni.models.seed_omni.modules.base.text_encoder.modeling.TextEncoder`
    decodes generated text via ``self._tokenizer``.
    """

    def __init__(self, chat_template: TextEncoderChatTemplate) -> None:
        self._chat_template = chat_template
        self._tokenizer = chat_template.tokenizer

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        """Build this module's chat template from its checkpoint subfolder."""
        del module_path, config_overrides
        raise NotImplementedError(f"{cls.__name__} must implement build_chat_template()")

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Build straight from the checkpoint dir — no model instance needed."""
        del kwargs
        return cls(cls.build_chat_template(module_path, config_overrides=config_overrides))

    def _tokenize_conversation_kwargs(self, inference: bool, **kwargs: Any) -> dict[str, Any]:
        """Extra kwargs forwarded to :meth:`TextEncoderChatTemplate.tokenize_conversation`."""
        del kwargs
        return {"add_generation_prompt": inference}

    def __call__(
        self,
        conversation_list: list[list[ConversationItem]],
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        tc_kwargs = self._tokenize_conversation_kwargs(inference, **kwargs)
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, **tc_kwargs)
            sample.clear()
            sample.extend(parts)


__all__ = ["TextEncoderPreprocessor"]
