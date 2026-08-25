"""Conversation packing helpers for BAGEL Qwen2-MoT."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....utils.conversation import _IMG_TAG_KEY, ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from .masking import build_mot_attention_metadata


_VALID_IMG_TAGS = frozenset({"und", "gen", "edit"})


@dataclass(frozen=True)
class PackedSpan:
    """Describe one logical span in the packed sequence.

    Most spans contain one carrier item. A VAE image span can group its leading
    marker, image, and trailing marker; ``primary_index`` identifies the image,
    while ``lengths`` preserves each item's subrange for routing and writeback.
    """

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
    """Result of packing an embedded conversation batch for MoT.

    Packed tensors concatenate all samples. ``sample_splits`` preserves their
    document/span boundaries for attention metadata. ``spans`` uses global
    packed offsets to map model outputs back to the original carrier items.
    Token type 0 selects the understanding expert and type 1 the generation
    expert.
    """

    packed_sequence: torch.Tensor
    sample_splits: list[list[int]]
    packed_attention_metadata: torch.Tensor
    packed_position_ids: torch.Tensor
    packed_token_type_ids: torch.Tensor
    spans: list[PackedSpan]


def preprocess_mot_inputs(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
) -> PackedConversation | None:
    """Preprocess already-embedded carrier items into MoT packed inputs.

    Text and SigLIP spans route through the understanding expert. VAE latent
    spans and output spans route through the generation expert. Attention mode
    and expert routing are deliberately computed separately: a clean VAE
    conditioning image still uses the generation expert. Packed layout stays
    local to MoT.
    """

    if not conversation_list:
        raise ValueError("BAGEL Qwen2-MoT preprocessing requires a non-empty conversation_list.")

    sequence_parts: list[torch.Tensor] = []
    sample_splits_by_sample: list[list[int]] = []
    sample_attn_modes_by_sample: list[list[str]] = []
    position_parts: list[torch.Tensor] = []
    token_type_parts: list[torch.Tensor] = []
    spans: list[PackedSpan] = []
    sequence_cursor = 0

    for sample in conversation_list:
        sample_start = sequence_cursor
        sample_splits: list[int] = []
        sample_attn_modes: list[str] = []
        sample_position_cursor = 0

        # Dummy carrier items are handled separately as zero-gradient anchors;
        # only real user/assistant embeddings participate in packed attention.
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
                # Treat <image_start>, image tokens, and <image_end> as one
                # logical attention span while preserving three writeback
                # ranges and routing only the image tokens to the gen expert.
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

            # sequence_cursor is global across the packed batch; positions
            # restart independently for each logical document.
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
            token_type_ids = torch.zeros(length, device=device, dtype=torch.long)
            if gen_token_indexes.numel() > 0:
                token_type_ids[(gen_token_indexes - indexes[0]).to(device=device, dtype=torch.long)] = 1
            token_type_parts.append(token_type_ids)

        sample_len = sequence_cursor - sample_start
        if sample_len > 0:
            sample_splits_by_sample.append(list(sample_splits))
            sample_attn_modes_by_sample.append(list(sample_attn_modes))

    if not sequence_parts:
        return None

    packed_sequence = torch.cat(sequence_parts, dim=0).to(device=device, dtype=dtype)
    return PackedConversation(
        packed_sequence=packed_sequence,
        sample_splits=sample_splits_by_sample,
        packed_attention_metadata=build_mot_attention_metadata(
            sample_splits_by_sample,
            sample_attn_modes_by_sample,
            device=device,
        ),
        packed_position_ids=torch.cat(position_parts, dim=0).to(device=device, dtype=torch.long),
        packed_token_type_ids=torch.cat(token_type_parts, dim=0).to(device=device, dtype=torch.long),
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
    """Assign BAGEL positions and return the next document-local position."""
    meta_position_ids = item.meta.get("position_ids")
    if torch.is_tensor(meta_position_ids):
        position_ids = meta_position_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
        if int(position_ids.numel()) != length:
            raise ValueError("BAGEL Qwen2-MoT position_ids length must match the item span length.")
        next_start = max(start + 1, int(position_ids.max().item()) + 1) if length else start
        return position_ids, next_start

    # All latent/image tokens in one span share a position; text remains AR
    # sequential. Thus a whole image advances the logical position only once.
    if item.type in {"image", "output"}:
        return torch.full((length,), start, device=device, dtype=torch.long), start + 1
    return torch.arange(start, start + length, device=device, dtype=torch.long), start + length


def _mot_attn_mode_for_item(item: ConversationItem) -> str:
    """Classify visibility independently from MoT expert routing."""
    if item.type == "output":
        return "noise"
    if item.type == "image":
        if item.source != BAGEL_VAE_CONTEXT:
            # SigLIP context (and any non-VAE image) is clean/full-attention
            # regardless of ``_img_tag``; the tag only disambiguates VAE roles.
            return "full"
        tag = item.meta.get(_IMG_TAG_KEY)
        if tag is None:
            # Untagged VAE images come from inference prompt routing. Training
            # targets must carry ``_img_tag="gen"``.
            return "full"
        if tag not in _VALID_IMG_TAGS:
            raise ValueError(f"BAGEL Qwen2-MoT received image with unknown {_IMG_TAG_KEY}: {tag!r}.")
        if tag == "gen":
            return "noise"
        if tag == "edit":
            # Edit VAE context is a clean conditioning image, not a noised target.
            return "full"
        raise ValueError(f"BAGEL Qwen2-MoT received VAE image with incompatible {_IMG_TAG_KEY}: {tag!r}.")
    return "causal"


def _mot_gen_token_indexes_for_span(span: PackedSpan, indexes: torch.Tensor) -> torch.Tensor:
    """Return packed rows routed through the generation expert."""
    item = span.item
    if item.type == "output":
        return indexes
    if item.type != "image" or item.source != BAGEL_VAE_CONTEXT:
        return indexes.new_empty(0)

    tag = item.meta.get(_IMG_TAG_KEY)
    if tag is not None and tag not in _VALID_IMG_TAGS:
        raise ValueError(f"BAGEL Qwen2-MoT received image with unknown {_IMG_TAG_KEY}: {tag!r}.")
    if tag == "und":
        raise ValueError(f"BAGEL Qwen2-MoT received VAE image with incompatible {_IMG_TAG_KEY}: {tag!r}.")

    # Expert routing follows token modality, independently of whether the VAE
    # latent is clean conditioning (untagged/edit) or a noised generation
    # target. Marker tokens remain on the understanding path.
    if span.is_image_triplet:
        primary_start = span.primary_start
        primary_end = primary_start + span.primary_length
        return indexes[primary_start:primary_end]
    return indexes


def _is_vision_marker_triplet_at(items: list[ConversationItem], index: int) -> bool:
    # Source equality prevents unrelated one-token text items around an image
    # from being mistaken for BAGEL's encoder-specific marker pair.
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


__all__ = [
    "PackedConversation",
    "PackedSpan",
    "preprocess_mot_inputs",
]
