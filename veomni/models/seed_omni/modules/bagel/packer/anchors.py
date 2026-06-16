"""BAGEL training-time zero-anchor helpers.

Several helpers here implement the missing-modality "zero anchor" strategy. Every
active SeedOmni V2 training node must run on every micro-batch so FSDP/DP collective
order stays aligned. When a sample lacks a modality, the owning node still touches its
parameters through a zero-valued anchor so those parameters appear in the autograd
graph (with zero gradient) instead of being dropped, which would otherwise desync
gradient reduction across ranks.
"""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem
from .carrier import conversation_samples


BAGEL_DUMMY_ANCHORS_META_KEY = "_bagel_dummy_anchors"


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
    # Add the zero-valued anchors into ``tensor`` so the upstream modules that produced
    # them stay connected to the loss graph without changing the numeric output.
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
