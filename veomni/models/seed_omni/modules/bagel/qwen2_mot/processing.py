"""Conversation packing helpers for BAGEL Qwen2-MoT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT


@dataclass(frozen=True)
class PackedSpan:
    start: int
    items: tuple[ConversationItem, ...]
    lengths: tuple[int, ...]
    primary_index: int = 0

    @property
    def item(self) -> ConversationItem:
        return self.items[self.primary_index]

    @property
    def length(self) -> int:
        return sum(self.lengths)

    @property
    def primary_start(self) -> int:
        return sum(self.lengths[: self.primary_index])

    @property
    def primary_length(self) -> int:
        return self.lengths[self.primary_index]

    @property
    def is_image_triplet(self) -> bool:
        return (
            len(self.items) == 3
            and self.primary_index == 1
            and self.items[0].type == "text"
            and self.items[1].type == "image"
            and self.items[2].type == "text"
        )


@dataclass
class PackedConversation:
    packed_sequence: torch.Tensor
    sample_lens: list[int]
    nested_attention_masks: list[torch.Tensor]
    packed_position_ids: torch.Tensor
    packed_und_token_indexes: torch.Tensor
    packed_gen_token_indexes: torch.Tensor
    spans: list[PackedSpan]


def preprocess_mot_inputs(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
) -> PackedConversation | None:
    """Preprocess already-embedded carrier items into MoT packed inputs.

    Text and image spans route through the understanding expert. Output spans
    route through the generation expert. Packed layout stays local to MoT.
    """

    if not conversation_list:
        return None

    sequence_parts: list[torch.Tensor] = []
    sample_lens: list[int] = []
    nested_attention_masks: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    und_indexes: list[torch.Tensor] = []
    gen_indexes: list[torch.Tensor] = []
    spans: list[PackedSpan] = []
    sequence_cursor = 0

    for sample in conversation_list:
        sample_start = sequence_cursor
        sample_splits: list[int] = []
        sample_attn_modes: list[str] = []
        sample_position_cursor = 0

        items = list(
            iter_desired_items(
                [sample],
                types=["text", "image", "output"],
                roles=["user", "assistant"],
            )
        )
        item_index = 0
        while item_index < len(items):
            if _is_vision_marker_triplet_at(items, item_index):
                span_items = (items[item_index], items[item_index + 1], items[item_index + 2])
                span_values = [
                    _mot_value_for_item(item, device=device, dtype=dtype, hidden_size=hidden_size)
                    for item in span_items
                ]
                span_lengths = tuple(int(value.shape[0]) for value in span_values)
                item = span_items[1]
                value = torch.cat(span_values, dim=0)
                length = int(value.shape[0])
                primary_index = 1
                item_index += 3
            else:
                item = items[item_index]
                value = _mot_value_for_item(item, device=device, dtype=dtype, hidden_size=hidden_size)
                length = int(value.shape[0])
                span_items = (item,)
                span_lengths = (length,)
                primary_index = 0
                item_index += 1

            if length == 0:
                continue

            indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=device, dtype=torch.long)
            position_ids, sample_position_cursor = _mot_position_ids_for_span(
                item,
                start=sample_position_cursor,
                length=length,
                device=device,
            )
            mode = _mot_attn_mode_for_item(item)

            sequence_parts.append(value)
            position_parts.append(position_ids)
            sample_splits.append(length)
            sample_attn_modes.append(mode)
            span = PackedSpan(
                start=sequence_cursor,
                items=span_items,
                lengths=span_lengths,
                primary_index=primary_index,
            )
            spans.append(span)
            sequence_cursor += length

            gen_token_indexes = _mot_gen_token_indexes_for_span(span, indexes)
            if gen_token_indexes.numel() > 0:
                gen_indexes.append(gen_token_indexes)
            if int(gen_token_indexes.numel()) < int(indexes.numel()):
                und_indexes.append(_index_difference(indexes, gen_token_indexes))

        sample_len = sequence_cursor - sample_start
        if sample_len > 0:
            sample_lens.append(sample_len)
            attention_mask = _mot_attn_mask_for_sample(sample_splits, sample_attn_modes)
            nested_attention_masks.append(attention_mask.to(device=device))

    if not sequence_parts:
        return None

    packed_sequence = torch.cat(sequence_parts, dim=0).to(device=device, dtype=dtype)
    return PackedConversation(
        packed_sequence=packed_sequence,
        sample_lens=sample_lens,
        nested_attention_masks=nested_attention_masks,
        packed_position_ids=torch.cat(position_parts, dim=0).to(device=device, dtype=torch.long),
        packed_und_token_indexes=(
            torch.cat(und_indexes, dim=0).to(device=device, dtype=torch.long)
            if und_indexes
            else torch.empty(0, device=device, dtype=torch.long)
        ),
        packed_gen_token_indexes=(
            torch.cat(gen_indexes, dim=0).to(device=device, dtype=torch.long)
            if gen_indexes
            else torch.empty(0, device=device, dtype=torch.long)
        ),
        spans=spans,
    )


def _mot_value_for_item(
    item: ConversationItem,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
) -> torch.Tensor:
    value = item.value
    if not torch.is_tensor(value):
        raise ValueError(f"BAGEL Qwen2-MoT expects embedded item tensors, got {type(value).__name__}.")
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(f"BAGEL Qwen2-MoT expects embedded item tensors, got shape {tuple(item.value.shape)}.")
    if int(value.shape[-1]) != int(hidden_size):
        raise ValueError(f"BAGEL Qwen2-MoT item hidden-size mismatch: got {value.shape[-1]}, expected {hidden_size}.")
    return value.to(device=device, dtype=dtype)


def _mot_position_ids_for_span(
    item: ConversationItem,
    *,
    start: int,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    meta_position_ids = item.meta.get("position_ids")
    if torch.is_tensor(meta_position_ids):
        position_ids = meta_position_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
        if int(position_ids.numel()) != length:
            raise ValueError("BAGEL Qwen2-MoT position_ids length must match the item span length.")
        next_start = max(start + 1, int(position_ids.max().item()) + 1) if length else start
        return position_ids, next_start

    if item.type in {"image", "output"}:
        return torch.full((length,), start, device=device, dtype=torch.long), start + 1
    return torch.arange(start, start + length, device=device, dtype=torch.long), start + length


def _mot_attn_mode_for_item(item: ConversationItem) -> str:
    if item.type == "output":
        return "noise"
    if item.type == "image":
        return "noise" if item.source == BAGEL_VAE_CONTEXT else "full"
    return "causal"


def _mot_gen_token_indexes_for_span(span: PackedSpan, indexes: torch.Tensor) -> torch.Tensor:
    item = span.item
    if item.type == "output":
        return indexes
    if item.type != "image" or item.source != BAGEL_VAE_CONTEXT:
        return indexes.new_empty(0)
    if span.is_image_triplet:
        primary_start = span.primary_start
        primary_end = primary_start + span.primary_length
        return indexes[primary_start:primary_end]
    return indexes


def _index_difference(indexes: torch.Tensor, remove: torch.Tensor) -> torch.Tensor:
    if remove.numel() == 0:
        return indexes
    keep = torch.ones(indexes.shape, device=indexes.device, dtype=torch.bool)
    start = int(indexes[0].item())
    keep[(remove - start).to(device=indexes.device, dtype=torch.long)] = False
    return indexes[keep]


def _is_vision_marker_triplet_at(items: list[ConversationItem], index: int) -> bool:
    if index + 2 >= len(items):
        return False
    start, image, end = items[index], items[index + 1], items[index + 2]
    if image.type != "image" or image.source not in {BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT}:
        return False
    return (
        start.source == image.source
        and end.source == image.source
        and _is_bagel_vision_marker(start, source=image.source)
        and _is_bagel_vision_marker(end, source=image.source)
    )


def _is_bagel_vision_marker(item: ConversationItem, *, source: str) -> bool:
    return item.type == "text" and item.source == source and _text_item_length(item) == 1


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


def _mot_attn_mask_for_sample(split_lens: Iterable[int], attn_modes: Iterable[str]) -> torch.Tensor:
    split_lens = list(split_lens)
    attn_modes = list(attn_modes)
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool)

    # Build the base packed-sequence visibility: every span can attend previous
    # context, while text spans keep causal visibility inside the span.
    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode == "causal":
            attention_mask[cursor : cursor + length, cursor : cursor + length] = torch.ones(
                (length, length), dtype=torch.bool
            ).tril()
        else:
            attention_mask[cursor : cursor + length, cursor : cursor + length] = True
        attention_mask[cursor : cursor + length, :cursor] = True
        cursor += length

    # Noise/output spans can attend themselves and previous context, but no
    # other span should see noise tokens as context.
    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode == "noise":
            attention_mask[:, cursor : cursor + length] = False
            attention_mask[cursor : cursor + length, cursor : cursor + length] = True
        cursor += length

    return torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(~attention_mask, float("-inf"))


__all__ = [
    "PackedConversation",
    "PackedSpan",
    "preprocess_mot_inputs",
]
