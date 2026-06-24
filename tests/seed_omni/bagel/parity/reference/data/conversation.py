"""BAGEL reference conversation conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from PIL import Image

from veomni.models.seed_omni.utils.conversation import ConversationItem, is_dummy


def conversation_item_text(
    item: ConversationItem,
    tokenizer: Any,
    *,
    input_ids_meta_key: str = "input_ids",
    skip_special_tokens: bool = True,
) -> str:
    """Convert a text conversation item back to a string for official BAGEL."""

    if isinstance(item.value, str):
        return item.value
    input_ids = item.meta.get(input_ids_meta_key, item.value)
    if torch.is_tensor(input_ids):
        return tokenizer.decode(
            input_ids.reshape(-1).detach().cpu().tolist(),
            skip_special_tokens=skip_special_tokens,
        )
    raise TypeError(f"BAGEL reference text input must be str or token tensor, got {type(item.value).__name__}.")


def conversation_item_image(item: ConversationItem) -> Image.Image:
    """Convert an image conversation item back to a PIL image for official BAGEL."""

    if isinstance(item.value, Image.Image):
        return item.value
    raise TypeError(f"BAGEL reference image input must be a PIL image, got {type(item.value).__name__}.")


def conversation_to_interleaved_reference_inputs(
    conversation: Iterable[ConversationItem],
    *,
    tokenizer: Any,
    skip_dummy: bool = True,
    input_ids_meta_key: str = "input_ids",
    skip_special_tokens: bool = True,
) -> list[str | Image.Image]:
    """Convert one V2 conversation sample to official interleaved inputs."""

    inputs: list[str | Image.Image] = []
    for item in conversation:
        if skip_dummy and is_dummy(item):
            continue
        if item.type == "text":
            inputs.append(
                conversation_item_text(
                    item,
                    tokenizer,
                    input_ids_meta_key=input_ids_meta_key,
                    skip_special_tokens=skip_special_tokens,
                )
            )
        elif item.type == "image":
            inputs.append(conversation_item_image(item))
        else:
            raise ValueError(f"BAGEL interleaved input does not support conversation item type {item.type!r}.")
    return inputs


__all__ = [
    "conversation_item_image",
    "conversation_item_text",
    "conversation_to_interleaved_reference_inputs",
]
