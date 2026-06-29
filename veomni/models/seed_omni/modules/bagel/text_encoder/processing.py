"""Stateless text carrier helpers for BAGEL text encoder."""

from __future__ import annotations

import copy

import torch
from PIL import Image

from ....utils.conversation import ConversationItem
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT


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


def copy_image_item(item: ConversationItem) -> ConversationItem:
    value = item.value
    if torch.is_tensor(value):
        value = value.clone()
    elif isinstance(value, Image.Image):
        value = value.copy()
    else:
        value = copy.deepcopy(value)
    return ConversationItem(
        type=item.type,
        value=value,
        role=item.role,
        source=item.source,
        meta=copy.deepcopy(item.meta),
    )


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


__all__ = [
    "apply_image_marker",
    "copy_image_item",
    "is_bagel_vision_marker",
]
