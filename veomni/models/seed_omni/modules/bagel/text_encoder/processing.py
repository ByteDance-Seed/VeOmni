"""Stateless text carrier helpers for BAGEL text encoder."""

from __future__ import annotations

import torch

from ....conversation import ConversationItem, is_dummy
from ..sources import BAGEL_FLOW_QUERY, BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT


IMAGE_MARKER_TARGETS = frozenset(
    {
        ("image", BAGEL_SIGLIP_CONTEXT),
        ("output", BAGEL_VAE_CONTEXT),
        ("output", BAGEL_FLOW_QUERY),
    }
)


def is_image_item(item: ConversationItem) -> bool:
    return not is_dummy(item) and (item.type, item.source) in IMAGE_MARKER_TARGETS


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
    "is_image_item",
]
