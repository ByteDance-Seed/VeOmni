"""BAGEL training-time carrier helpers."""

from __future__ import annotations

from typing import Any

import torch

from ...conversation import ConversationItem


BAGEL_PACKED_BATCH_META_KEY = "_bagel_packed_batch"
BAGEL_DUMMY_ANCHORS_META_KEY = "_bagel_dummy_anchors"


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


def add_dummy_anchor_to_batch(batch: dict[str, Any], anchor: torch.Tensor) -> None:
    anchors = batch.setdefault(BAGEL_DUMMY_ANCHORS_META_KEY, [])
    if not isinstance(anchors, list):
        raise TypeError(f"{BAGEL_DUMMY_ANCHORS_META_KEY} must be a list.")
    anchors.append(anchor)


def append_dummy_anchor(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
    anchor: torch.Tensor,
) -> None:
    samples = conversation_samples(conversation_list)
    if not samples or not samples[0]:
        return
    anchors = samples[0][0].meta.setdefault(BAGEL_DUMMY_ANCHORS_META_KEY, [])
    if not isinstance(anchors, list):
        raise TypeError(f"{BAGEL_DUMMY_ANCHORS_META_KEY} must be a list.")
    anchors.append(anchor)


def dummy_anchors_from_conversation(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> list[torch.Tensor]:
    anchors: list[torch.Tensor] = []
    for sample in conversation_samples(conversation_list):
        for item in sample:
            value = item.meta.get(BAGEL_DUMMY_ANCHORS_META_KEY)
            if isinstance(value, list):
                anchors.extend(anchor for anchor in value if torch.is_tensor(anchor))
    return anchors


def fold_dummy_anchors(tensor: torch.Tensor, anchors: list[torch.Tensor]) -> torch.Tensor:
    if not anchors:
        return tensor
    anchor = tensor.new_zeros(())
    for value in anchors:
        if torch.is_tensor(value):
            anchor = anchor + value.to(device=tensor.device, dtype=tensor.dtype)
    return tensor + anchor


def zero_hidden_from_batch(
    batch: dict[str, Any],
    *,
    hidden_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden_states = batch.get("packed_hidden_states")
    if torch.is_tensor(hidden_states):
        return hidden_states[:1].to(device=device, dtype=dtype) * 0.0
    return torch.zeros(1, hidden_size, device=device, dtype=dtype)


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


__all__ = [
    "BAGEL_DUMMY_ANCHORS_META_KEY",
    "BAGEL_PACKED_BATCH_META_KEY",
    "add_dummy_anchor_to_batch",
    "clear_packed_batch",
    "conversation_samples",
    "dummy_anchors_from_conversation",
    "fold_dummy_anchors",
    "get_packed_batch",
    "append_dummy_anchor",
    "packed_label_rows",
    "require_packed_batch",
    "set_packed_batch",
    "zero_hidden_from_batch",
]
