"""Worker-side preprocessor for :class:`Qwen3TextEncoder` — HF ``AutoProcessor`` style.

:class:`Qwen3TextEncoderPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.mixins.modulemixin.Preprocessor`).
Its :meth:`from_pretrained` loads the tokenizer and picks the chat template
straight off the checkpoint dir — no model instance involved. The template
choice mirrors the module's own ``config.enable_image`` flag: image mode reuses
the Qwen3-VL ChatML template (adds the vision wrap tokens); otherwise the
text-only Qwen3 ChatML.
"""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ....mixins.modulemixin import Preprocessor
from ....utils.conversation import ConversationItem
from ...qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from .chat_template import Qwen3ChatTemplate
from .configuration import Qwen3TextEncoderConfig


class Qwen3TextEncoderPreprocessor(Preprocessor):
    """Worker-side ``apply_chat_template`` → tokenize → merge for the Qwen3 text encoder.

    Holds only the (picklable) chat template (text-only or the reused Qwen3-VL image
    template) — never the model — so it runs in DataLoader workers and overlaps with
    GPU compute. Builds CPU tensors; the main process's thin ``encode_pre`` does the
    single ``.to(device)``.
    """

    def __init__(self, chat_template: Qwen3ChatTemplate | Qwen3VLChatTemplate) -> None:
        self._chat_template = chat_template
        # bind_preprocessor also copies _tokenizer onto the model — TextEncoderModuleMixin
        # (base/text_encoder/modulemixin.py) decodes generated text via self._tokenizer.
        self._tokenizer = chat_template.tokenizer

    @classmethod
    def from_pretrained(
        cls, module_path: str, *, config_overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> Qwen3TextEncoderPreprocessor:
        """Build straight from the checkpoint dir — no model instance needed.

        ``config_overrides`` (the module's YAML ``model_config:`` block, e.g.
        visual-instruction-tuning's ``enable_image: true``) is applied on top of
        the on-disk ``config.json`` default so the template choice below agrees
        with what the live model was actually configured with.
        """
        del kwargs
        config = Qwen3TextEncoderConfig.from_pretrained(module_path, **(config_overrides or {}))
        tokenizer = build_tokenizer(module_path)
        chat_template = Qwen3VLChatTemplate(tokenizer) if config.enable_image else Qwen3ChatTemplate(tokenizer)
        return cls(chat_template)

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list or []:
            parts = self._chat_template.tokenize_conversation(sample, add_generation_prompt=inference)
            sample.clear()
            sample.extend(parts)


__all__ = ["Qwen3TextEncoderPreprocessor"]
