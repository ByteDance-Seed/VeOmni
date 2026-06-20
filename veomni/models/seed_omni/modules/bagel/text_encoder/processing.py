"""Stateless text carrier helpers for BAGEL text encoder."""

from __future__ import annotations

from typing import Any

import torch

from veomni.utils.tensor_utils import naflatten

from ....conversation import ConversationItem, is_dummy, maybe_merge_outputs, seal_outputs


def resolve_token_id(tokenizer: Any, token: str, fallback: int | None) -> int | None:
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return fallback
    resolved = tokenizer.convert_tokens_to_ids(token)
    if resolved is None:
        return fallback
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and int(resolved) == int(unk):
        return fallback
    return int(resolved)


def text_item_input_ids(
    item: ConversationItem,
    *,
    tokenizer: Any,
    start_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    meta_ids = item.meta.get("input_ids")
    if torch.is_tensor(meta_ids):
        return meta_ids.detach().to(device=device, dtype=torch.long).reshape(-1)

    value = item.value
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.long).reshape(-1)

    if not isinstance(value, str):
        raise TypeError(f"BAGEL text item value must be str or token tensor, got {type(value).__name__}.")
    if tokenizer is None:
        raise ValueError("BAGEL text encoder requires a tokenizer for raw string input.")

    token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    return torch.tensor([start_token_id, *token_ids, eos_token_id], device=device, dtype=torch.long)


def labels_for_text_item(item: ConversationItem, token_ids: torch.Tensor) -> torch.Tensor:
    if item.role == "assistant":
        return token_ids.detach().clone()
    return torch.full_like(token_ids, -100, dtype=torch.long)


def text_hidden_and_labels(
    item: ConversationItem,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    hidden = item.value
    labels = item.meta.get("labels")
    if not torch.is_tensor(hidden) or not torch.is_tensor(labels):
        return None
    if hidden.dim() == 3 and hidden.size(0) == 1:
        hidden = hidden.squeeze(0)
    hidden = hidden.to(device=device, dtype=dtype)
    labels = labels.to(device=device, dtype=torch.long).reshape(-1)
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError(
            "BAGEL text decode requires hidden-state and label lengths to match: "
            f"got {hidden.shape[0]} and {labels.shape[0]}."
        )
    return hidden, labels


def prepare_text_encode_inputs(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    tokenizer: Any,
    start_token_id: int,
    eos_token_id: int,
    device: torch.device,
    dummy_input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    input_ids: list[torch.Tensor] = []
    segment_count = 0
    for sample in conversation_list or []:
        for item in sample:
            if is_dummy(item) or item.type != "text":
                continue
            token_ids = text_item_input_ids(
                item,
                tokenizer=tokenizer,
                start_token_id=start_token_id,
                eos_token_id=eos_token_id,
                device=device,
            )
            item.value = token_ids
            item.meta["input_ids"] = token_ids.detach()
            item.meta["labels"] = labels_for_text_item(item, token_ids)
            item.meta["attention_mask"] = torch.ones_like(token_ids, dtype=torch.long)
            input_ids.append(token_ids)
            segment_count += 1

    if not input_ids:
        input_ids = [dummy_input_ids]
    flat_input_ids, batch_shape = naflatten(input_ids)
    return flat_input_ids, batch_shape, segment_count


def prepare_text_decode_inputs(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy_hidden_states: torch.Tensor,
    dummy_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for sample in conversation_list or []:
        for item in sample:
            if is_dummy(item) or item.type != "text":
                continue
            pair = text_hidden_and_labels(item, device=device, dtype=dtype)
            if pair is None:
                continue
            hidden, labels = pair
            hidden_parts.append(hidden)
            label_parts.append(labels)

    if not hidden_parts:
        return dummy_hidden_states, dummy_labels
    return torch.cat(hidden_parts, dim=0), torch.cat(label_parts, dim=0)


def scatter_text_embeds(
    conversation_list: list[list[ConversationItem]],
    segment_embeds: list[torch.Tensor],
    *,
    expected: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if expected == 0:
        return

    consumed = 0
    for sample in conversation_list:
        for item in sample:
            if is_dummy(item) or item.type != "text":
                continue
            if consumed >= len(segment_embeds):
                raise RuntimeError("BAGEL text segment count mismatch during embed scatter.")
            item.value = segment_embeds[consumed].to(device=device, dtype=dtype)
            consumed += 1
    if consumed != expected or consumed != len(segment_embeds):
        raise RuntimeError("BAGEL text segment count mismatch during embed scatter.")


def image_embed_marker_items(
    conversation_list: list[list[ConversationItem]],
    *,
    item_types: set[str],
) -> list[ConversationItem]:
    image_items: list[ConversationItem] = []
    for sample in conversation_list:
        for item in sample:
            if is_dummy(item) or item.type not in item_types:
                continue
            value = squeeze_single_batch(item.value)
            if not torch.is_tensor(value) or value.dim() != 2:
                continue
            image_items.append(item)
    return image_items


def squeeze_single_batch(value: object) -> object:
    if torch.is_tensor(value) and value.dim() == 3 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def apply_image_embed_markers(
    image_items: list[ConversationItem],
    marker_embeds: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if marker_embeds.dim() == 2:
        marker_embeds = marker_embeds.unsqueeze(0)
    for item, embeds in zip(image_items, marker_embeds, strict=True):
        image_embeds = squeeze_single_batch(item.value)
        if not torch.is_tensor(image_embeds):
            continue
        image_embeds = image_embeds.to(device=device, dtype=dtype)
        if (
            image_embeds.dim() == 2
            and image_embeds.shape[0] >= 2
            and torch.equal(image_embeds[:1], embeds[:1])
            and torch.equal(image_embeds[-1:], embeds[1:])
        ):
            continue
        item.value = torch.cat([embeds[:1], image_embeds, embeds[1:]], dim=0)


def output_hidden_tail(sample: list[ConversationItem]) -> tuple[ConversationItem, torch.Tensor]:
    if not sample:
        raise ValueError("BAGEL text token_generate requires a non-empty conversation_list.")
    tail = sample[-1]
    if tail.type != "output":
        raise ValueError(f"BAGEL text token_generate expects tail type 'output', got {tail.type!r}.")
    if not torch.is_tensor(tail.value):
        raise TypeError("BAGEL text token_generate expects the tail output value to be a hidden-state tensor.")
    hidden_states = tail.value
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    return tail, hidden_states


def sampled_token_id(token: int | torch.Tensor) -> int:
    if torch.is_tensor(token):
        return int(token.reshape(-1)[0].item())
    return int(token)


def update_tail_with_generated_token(
    sample: list[ConversationItem],
    tail: ConversationItem,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    tail.value = inputs_embeds.to(device=device, dtype=dtype)
    tail.meta["input_ids"] = input_ids.reshape(-1).detach()
    maybe_merge_outputs(sample)


def as_batched_inference_conversation(
    conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
) -> list[list[ConversationItem]]:
    if conversation_list is None:
        raise ValueError("BAGEL text inference endpoints require conversation_list.")
    if not conversation_list:
        return []
    first = conversation_list[0]
    if isinstance(first, ConversationItem):
        return [conversation_list]  # type: ignore[list-item]
    return conversation_list  # type: ignore[return-value]


def seal_text_output_span(conversation_list: list[ConversationItem] | list[list[ConversationItem]]) -> None:
    for sample in as_batched_inference_conversation(conversation_list):
        if not sample or sample[-1].type != "output":
            continue
        if len(sample) >= 2 and sample[-2].type == "output":
            sample.pop()
        if sample and sample[-1].type == "output":
            seal_outputs(sample, new_type="text")


def build_generated_text(
    conversation_list: list[ConversationItem] | list[list[ConversationItem]],
    *,
    tokenizer: Any,
    token_ids: list[int],
) -> dict[str, Any]:
    if not token_ids:
        return {}
    seal_text_output_span(conversation_list)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return {"type": "text", "value": text, "meta": {"token_ids": token_ids}}


__all__ = [
    "apply_image_embed_markers",
    "as_batched_inference_conversation",
    "build_generated_text",
    "image_embed_marker_items",
    "labels_for_text_item",
    "output_hidden_tail",
    "prepare_text_decode_inputs",
    "prepare_text_encode_inputs",
    "resolve_token_id",
    "sampled_token_id",
    "scatter_text_embeds",
    "seal_text_output_span",
    "squeeze_single_batch",
    "text_hidden_and_labels",
    "text_item_input_ids",
    "update_tail_with_generated_token",
]
