"""Stateless text carrier helpers for BAGEL text encoder."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from ....conversation import ConversationItem, is_dummy


def materialize_text_item_input_ids(
    item: ConversationItem,
    tokenizer: PreTrainedTokenizerBase,
    *,
    start_token_id: int,
    eos_token_id: int,
    tokenized_key: str | None = None,
) -> torch.Tensor | None:
    if is_dummy(item) or item.type != "text":
        return None
    if tokenized_key is not None and item.meta.get(tokenized_key):
        return item.meta["input_ids"]

    token_ids = tokenizer(item.value, add_special_tokens=False)["input_ids"]
    token_ids = torch.tensor([start_token_id, *token_ids, eos_token_id], dtype=torch.long)
    item.value = token_ids
    item.meta["input_ids"] = token_ids.detach()
    item.meta["attention_mask"] = torch.ones_like(token_ids, dtype=torch.long)
    item.meta["labels"] = (
        token_ids.detach().clone() if item.role == "assistant" else torch.full_like(token_ids, -100, dtype=torch.long)
    )
    if tokenized_key is not None:
        item.meta[tokenized_key] = True
    return token_ids


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
    "materialize_text_item_input_ids",
]
