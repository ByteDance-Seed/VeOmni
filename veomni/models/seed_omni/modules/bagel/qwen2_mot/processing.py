"""Conversation packing helpers for BAGEL Qwen2-MoT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ....conversation import ConversationItem, is_dummy


def active_output_item(conversation: list[ConversationItem]) -> ConversationItem | None:
    for item in reversed(conversation):
        if item.type == "output":
            return item
    return None


def validate_cfg_request(generation_kwargs: dict[str, object]) -> None:
    cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
    cfg_img_scale = float(generation_kwargs.get("cfg_img_scale", 1.0))
    if cfg_img_scale > 1.0 and cfg_text_scale <= 1.0:
        raise ValueError("cfg_img_scale > 1.0 requires cfg_text_scale > 1.0")


def cfg_text_active(generation_kwargs: dict[str, object], timestep: object) -> bool:
    cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
    if cfg_text_scale <= 1.0:
        return False
    lower, upper = cfg_interval(generation_kwargs.get("cfg_interval", (0.0, 1.0)))
    t = timestep_value(timestep)
    return t > lower and t <= upper


def cfg_img_active(generation_kwargs: dict[str, object], timestep: object) -> bool:
    validate_cfg_request(generation_kwargs)
    cfg_img_scale = float(generation_kwargs.get("cfg_img_scale", 1.0))
    return cfg_img_scale > 1.0 and cfg_text_active(generation_kwargs, timestep)


def cfg_branch_sequence(generation_kwargs: dict[str, object], timestep: object) -> tuple[str, ...]:
    branches = ["main"]
    if cfg_text_active(generation_kwargs, timestep):
        branches.append("cfg_text")
    if cfg_img_active(generation_kwargs, timestep):
        branches.append("cfg_img")
    return tuple(branches)


def cfg_interval(value: object) -> tuple[float, float]:
    if value is None:
        return 0.0, 1.0
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, 1.0


def timestep_value(value: object) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("BAGEL CFG timestep tensor must not be empty.")
        return float(value.detach().reshape(-1)[0].item())
    if value is None:
        raise ValueError("BAGEL CFG requires current-round meta['timestep'].")
    return float(value)


def merge_cfg_velocity(
    *,
    main_velocity: torch.Tensor,
    cfg_text_velocity: torch.Tensor,
    cfg_img_velocity: torch.Tensor | None = None,
    generation_kwargs: dict[str, object],
) -> torch.Tensor:
    cfg_text_scale = float(generation_kwargs.get("cfg_text_scale", 1.0))
    cfg_img_scale = float(generation_kwargs.get("cfg_img_scale", 1.0))
    cfg_renorm_min = float(generation_kwargs.get("cfg_renorm_min", 0.0))
    cfg_renorm_type = str(generation_kwargs.get("cfg_renorm_type", "global"))
    if cfg_img_scale > 1.0 and cfg_img_velocity is None:
        raise ValueError("BAGEL image CFG merge requires a collected cfg_img velocity.")
    if cfg_text_scale <= 1.0:
        return main_velocity

    text_guided = cfg_text_velocity + cfg_text_scale * (main_velocity - cfg_text_velocity)
    if cfg_renorm_type == "text_channel":
        norm_main = torch.norm(main_velocity, dim=-1, keepdim=True)
        norm_text = torch.norm(text_guided, dim=-1, keepdim=True)
        scale = (norm_main / (norm_text + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        guided = text_guided * scale
        if cfg_img_scale > 1.0:
            assert cfg_img_velocity is not None
            guided = cfg_img_velocity + cfg_img_scale * (guided - cfg_img_velocity)
        return guided

    if cfg_img_scale > 1.0:
        assert cfg_img_velocity is not None
        guided = cfg_img_velocity + cfg_img_scale * (text_guided - cfg_img_velocity)
    else:
        guided = text_guided
    if cfg_renorm_type == "global":
        norm_main = torch.norm(main_velocity)
        norm_guided = torch.norm(guided)
    elif cfg_renorm_type == "channel":
        norm_main = torch.norm(main_velocity, dim=-1, keepdim=True)
        norm_guided = torch.norm(guided, dim=-1, keepdim=True)
    else:
        raise NotImplementedError(f"BAGEL infer_gen CFG renorm type {cfg_renorm_type!r} is not implemented.")
    scale = (norm_main / (norm_guided + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
    return guided * scale


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


def pack_training_conversation(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
) -> PackedConversation | None:
    """Pack already-embedded carrier items in sample order.

    Text and image spans route through the understanding expert. Output spans
    route through the generation expert. Packed layout stays local to MoT.
    """

    samples = conversation_list or []
    sequence_parts: list[torch.Tensor] = []
    sample_lens: list[int] = []
    nested_attention_masks: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    und_indexes: list[torch.Tensor] = []
    gen_indexes: list[torch.Tensor] = []
    spans: list[PackedSpan] = []
    sequence_cursor = 0

    for sample in samples:
        sample_start = sequence_cursor
        sample_splits: list[int] = []
        sample_attn_modes: list[str] = []
        sample_position_cursor = 0

        for item in sample:
            if is_dummy(item) or item.type not in {"text", "image", "output"}:
                continue
            value = item_tensor_value(item, device=device, dtype=dtype, hidden_size=hidden_size)
            if value is None:
                continue
            length = int(value.shape[0])
            if length == 0:
                continue

            indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=device, dtype=torch.long)
            position_ids, sample_position_cursor = position_ids_for_item(
                item,
                start=sample_position_cursor,
                length=length,
                device=device,
            )
            mode = attention_mode_for_item(item)

            sequence_parts.append(value)
            position_parts.append(position_ids)
            sample_splits.append(length)
            sample_attn_modes.append(mode)
            spans.append(PackedSpan(item=item, start=sequence_cursor, length=length))
            if item.type == "output":
                gen_indexes.append(indexes)
            else:
                und_indexes.append(indexes)
            sequence_cursor += length

        sample_len = sequence_cursor - sample_start
        if sample_len > 0:
            sample_lens.append(sample_len)
            nested_attention_masks.append(
                prepare_attention_mask_per_sample(sample_splits, sample_attn_modes).to(device=device)
            )

    if not sequence_parts:
        return None

    packed_sequence = torch.cat(sequence_parts, dim=0).to(device=device, dtype=dtype)
    if int(packed_sequence.shape[-1]) != hidden_size:
        raise ValueError(
            f"BAGEL Qwen2-MoT hidden-size mismatch: got {packed_sequence.shape[-1]}, expected {hidden_size}."
        )
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


def item_tensor_value(
    item: ConversationItem,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
) -> torch.Tensor | None:
    value = item.value
    if not torch.is_tensor(value):
        return None
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(f"BAGEL Qwen2-MoT expects embedded item tensors, got shape {tuple(item.value.shape)}.")
    if int(value.shape[-1]) != int(hidden_size):
        raise ValueError(f"BAGEL Qwen2-MoT item hidden-size mismatch: got {value.shape[-1]}, expected {hidden_size}.")
    return value.to(device=device, dtype=dtype)


def position_ids_for_item(
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


def attention_mode_for_item(item: ConversationItem) -> str:
    if item.type == "image":
        return "full"
    if item.type == "output":
        return "noise"
    return "causal"


def prepare_attention_mask_per_sample(split_lens: Iterable[int], attn_modes: Iterable[str]) -> torch.Tensor:
    split_lens = list(split_lens)
    attn_modes = list(attn_modes)
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool)
    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode not in {"causal", "full", "noise"}:
            raise ValueError(f"Unsupported BAGEL Qwen2-MoT attention mode: {mode!r}")
        if mode == "causal":
            attention_mask[cursor : cursor + length, cursor : cursor + length] = torch.ones(
                (length, length), dtype=torch.bool
            ).tril()
        else:
            attention_mask[cursor : cursor + length, cursor : cursor + length] = True
        attention_mask[cursor : cursor + length, :cursor] = True
        cursor += length

    cursor = 0
    for length, mode in zip(split_lens, attn_modes, strict=True):
        if mode == "noise":
            attention_mask[:, cursor : cursor + length] = False
            attention_mask[cursor : cursor + length, cursor : cursor + length] = True
        cursor += length
    return torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(~attention_mask, float("-inf"))


def scatter_hidden_states(spans: list[PackedSpan], hidden_states: torch.Tensor, *, device: torch.device) -> None:
    for span in spans:
        span.item.value = hidden_states[span.start : span.start + span.length].to(device=device)


def single_inference_conversation(
    conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
) -> list[ConversationItem]:
    if conversation_list is None:
        raise ValueError("BAGEL Qwen2-MoT generate requires conversation_list.")
    if not conversation_list:
        raise ValueError("BAGEL Qwen2-MoT generate received an empty conversation_list.")
    first = conversation_list[0]
    if isinstance(first, ConversationItem):
        return conversation_list  # type: ignore[return-value]
    if isinstance(first, list):
        if len(conversation_list) != 1:
            raise ValueError("BAGEL Qwen2-MoT generate currently supports one inference sample at a time.")
        return first
    raise TypeError("BAGEL Qwen2-MoT conversation_list must contain ConversationItem objects.")


def tail_hidden_from_packed(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.dim() != 2:
        raise ValueError(f"BAGEL Qwen2-MoT expected packed hidden states, got {tuple(hidden_states.shape)}.")
    return hidden_states[-1:].contiguous()


def tail_query_embedding(item: ConversationItem) -> torch.Tensor:
    value = item.value
    if not torch.is_tensor(value):
        raise ValueError("BAGEL Qwen2-MoT decode expects tail output.value to be an embedding tensor.")
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(f"BAGEL Qwen2-MoT expected tail output embedding rank 2, got {tuple(value.shape)}.")
    return value[-1:].contiguous()


def next_position_ids_from_packed(packed: PackedConversation) -> torch.Tensor:
    next_positions: list[torch.Tensor] = []
    cursor = 0
    for sample_len in packed.sample_lens:
        sample_positions = packed.packed_position_ids[cursor : cursor + sample_len]
        next_positions.append(sample_positions.max().reshape(1) + 1)
        cursor += sample_len
    return torch.cat(next_positions, dim=0).to(device=packed.packed_position_ids.device, dtype=torch.long)


def shift_packed_key_value_indexes(
    packed_key_value_indexes: torch.Tensor,
    key_values_lens: torch.Tensor,
) -> torch.Tensor:
    parts = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
    for index, part in enumerate(parts):
        parts[index] = part + index
    return torch.cat(parts, dim=0)


def append_query_indexes_to_cache(
    packed_key_value_indexes: torch.Tensor,
    key_values_lens: torch.Tensor,
) -> torch.Tensor:
    parts = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
    for index, part in enumerate(parts):
        next_index = part[-1:] + 1
        parts[index] = torch.cat([part, next_index], dim=0)
    return torch.cat(parts, dim=0)


__all__ = [
    "PackedConversation",
    "PackedSpan",
    "active_output_item",
    "append_query_indexes_to_cache",
    "cfg_branch_sequence",
    "cfg_img_active",
    "cfg_text_active",
    "merge_cfg_velocity",
    "next_position_ids_from_packed",
    "pack_training_conversation",
    "prepare_attention_mask_per_sample",
    "scatter_hidden_states",
    "shift_packed_key_value_indexes",
    "single_inference_conversation",
    "tail_hidden_from_packed",
    "tail_query_embedding",
    "validate_cfg_request",
]
