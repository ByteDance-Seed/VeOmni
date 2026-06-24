"""Conversation packing helpers for BAGEL Qwen2-MoT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ....utils.conversation import ConversationItem, iter_desired_items


@dataclass(frozen=True)
class PackedSpan:
    item: ConversationItem
    start: int
    length: int


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

        for item in iter_desired_items(
            [sample],
            types=["text", "image", "output"],
            roles=["user", "assistant"],
        ):
            value = _mot_value_for_item(item, device=device, dtype=dtype, hidden_size=hidden_size)
            length = int(value.shape[0])
            if length == 0:
                continue

            indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=device, dtype=torch.long)
            position_ids, sample_position_cursor = _mot_position_ids_for_item(
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
            spans.append(PackedSpan(item=item, start=sequence_cursor, length=length))
            sequence_cursor += length
            if item.type == "output":
                gen_indexes.append(indexes)
            else:
                und_indexes.append(indexes)

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


def _mot_position_ids_for_item(
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
    if item.type == "image":
        return "full"
    if item.type == "output":
        return "noise"
    return "causal"


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
