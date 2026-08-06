"""Stateless text carrier helpers + worker-side preprocessor for BAGEL text encoder.

:class:`BagelTextEncoderPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.processing.base.ModulePreprocessorBase`).
Its :meth:`from_pretrained` loads the tokenizer and builds the
:class:`~veomni.models.seed_omni.modules.bagel.text_encoder.chat_template.BagelChatTemplate`
straight off the checkpoint dir — no model instance involved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .....auto import build_tokenizer
from ....utils.conversation import ConversationItem
from ...base.text_encoder.processing import TextEncoderPreprocessor
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT


if TYPE_CHECKING:
    # chat_template.py imports is_bagel_vision_marker from this module, so the
    # reverse import must stay deferred (see from_pretrained) to avoid a cycle.
    from .chat_template import BagelChatTemplate


def is_bagel_vision_marker(item: ConversationItem, *, source: str | None = None) -> bool:
    if item.type != "text":
        return False
    if source is not None and item.source != source:
        return False
    if item.source not in {BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT}:
        return False
    return _text_item_length(item) == 1


def _text_item_length(item: ConversationItem) -> int | None:
    value = item.value
    if torch.is_tensor(value):
        if value.dim() == 0:
            return 1
        if value.dim() == 1:
            return int(value.shape[0])
        if value.dim() == 2:
            return int(value.shape[0])
        if value.dim() == 3 and int(value.shape[0]) == 1:
            return int(value.shape[1])
        return None
    input_ids = item.meta.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.reshape(-1).shape[0])
    return None


def apply_image_marker(
    item: ConversationItem,
    marker_embeds: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    image_embeds = item.value
    if not torch.is_tensor(image_embeds):
        return
    if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
        image_embeds = image_embeds.squeeze(0)
    if image_embeds.dim() != 2:
        return

    image_embeds = image_embeds.to(device=device, dtype=dtype)
    if (
        image_embeds.shape[0] >= 2
        and torch.equal(image_embeds[:1], marker_embeds[:1])
        and torch.equal(image_embeds[-1:], marker_embeds[1:])
    ):
        return
    item.value = torch.cat([marker_embeds[:1], image_embeds, marker_embeds[1:]], dim=0)


class BagelTextEncoderPreprocessor(TextEncoderPreprocessor):
    """Worker-side chat-template + tokenize for BAGEL text encoder inputs."""

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> BagelChatTemplate:
        del config_overrides
        from .chat_template import BagelChatTemplate  # deferred: see module-level note

        return BagelChatTemplate(build_tokenizer(module_path))

    def _tokenize_conversation_kwargs(self, inference: bool, **kwargs: Any) -> dict[str, Any]:
        return {
            "add_generation_prompt": inference,
            "generation_kwargs": kwargs.get("generation_kwargs"),
        }

    def __call__(
        self,
        conversation_list: list[list[ConversationItem]],
        *,
        inference: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__call__(conversation_list, inference=inference, generation_kwargs=generation_kwargs)


__all__ = [
    "apply_image_marker",
    "is_bagel_vision_marker",
    "BagelTextEncoderPreprocessor",
]
