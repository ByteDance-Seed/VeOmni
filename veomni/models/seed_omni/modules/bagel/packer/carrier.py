"""BAGEL packed-batch carrier protocol.

Helpers for storing and retrieving the packed training batch on
``ConversationItem.meta`` across the SeedOmni V2 graph.
"""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem


BAGEL_PACKED_BATCH_META_KEY = "_bagel_packed_batch"


def conversation_samples(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> list[list[ConversationItem]]:
    if conversation_list is None:
        return []
    if not conversation_list:
        return []
    first = conversation_list[0]
    if isinstance(first, ConversationItem):
        return [conversation_list]  # type: ignore[list-item]
    return conversation_list  # type: ignore[return-value]


def get_packed_batch(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> dict[str, Any] | None:
    for sample in conversation_samples(conversation_list):
        for item in sample:
            packed = item.meta.get(BAGEL_PACKED_BATCH_META_KEY)
            if isinstance(packed, dict):
                return packed
    return None


def set_packed_batch(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
    packed_batch: dict[str, Any],
) -> None:
    samples = conversation_samples(conversation_list)
    if not samples or not samples[0]:
        raise ValueError("BAGEL training pack requires at least one ConversationItem.")
    samples[0][0].meta[BAGEL_PACKED_BATCH_META_KEY] = packed_batch


def require_packed_batch(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> dict[str, Any]:
    packed_batch = get_packed_batch(conversation_list)
    if packed_batch is None:
        raise ValueError("BAGEL packed training carrier is missing from conversation_list.")
    return packed_batch


def packed_label_rows(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> torch.Tensor:
    packed_batch = require_packed_batch(conversation_list)
    return torch.unique(packed_batch["packed_label_ids"].detach().cpu()).to(dtype=torch.long)


def clear_packed_batch(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> None:
    for sample in conversation_samples(conversation_list):
        for item in sample:
            item.meta.pop(BAGEL_PACKED_BATCH_META_KEY, None)
