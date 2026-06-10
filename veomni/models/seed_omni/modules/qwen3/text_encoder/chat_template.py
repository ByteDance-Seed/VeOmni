"""Qwen3 ChatML template as readable Python.

Training flow
-------------
1. :func:`apply_qwen3_chat_template` — wrap each turn in
   ``<|im_start|>{role}\\n{content}`` plus a trailing ``\\n``; set
   ``meta["loss_mask"]`` on assistant rows.
2. Text encoder tokenizes each text row, merges adjacent text, then packs.
3. :func:`pack_text_input_ids` — ``list[Tensor]`` of per-text-row token ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ....conversation import ConversationItem


@dataclass(frozen=True)
class Qwen3ChatMarkers:
    """Wire-format strings resolved from the module tokenizer."""

    im_start_token: str
    im_end_token: str
    eos_token: str
    assistant_prefix: str


def _template_item(
    item_type: str,
    value: Any,
    role: str,
    *,
    loss_mask: int | None = None,
    meta: dict | None = None,
) -> ConversationItem:
    part_meta = dict(meta or {})
    if item_type == "text":
        part_meta["loss_mask"] = int(role == "assistant") if loss_mask is None else int(loss_mask)
    return ConversationItem(type=item_type, value=value, role=role, meta=part_meta)


def apply_qwen3_chat_template(
    sample: list[ConversationItem],
    markers: Qwen3ChatMarkers,
) -> list[ConversationItem]:
    """Apply ChatML template to a raw text-only conversation."""
    out: list[ConversationItem] = []
    dummy_parts: list[ConversationItem] = []
    prev_role: str | None = None

    for item in sample:
        role = item.role
        if role == "dummy":
            dummy_parts.append(item)
            continue
        if item.type != "text":
            raise ValueError(f"Qwen3 text encoder only supports text items, got {item.type!r}")

        if role != prev_role:
            out.append(_template_item("text", markers.im_start_token + role + "\n", role, loss_mask=0))
            prev_role = role

        out.append(_template_item("text", str(item.value), role))

    if prev_role == "assistant":
        out.append(_template_item("text", markers.im_end_token + "\n", "assistant"))
    out.extend(dummy_parts)
    return out


def apply_qwen3_generation_prompt(
    sample: list[ConversationItem],
    markers: Qwen3ChatMarkers,
) -> list[ConversationItem]:
    """Close the last user turn and append the assistant generation prefix."""
    out = list(sample)
    if out and out[-1].role == "user" and out[-1].type == "text":
        out.append(_template_item("text", markers.im_end_token + "\n", "user", loss_mask=0))
    out.append(_template_item("text", markers.assistant_prefix, "assistant", loss_mask=0))
    return out


def pack_text_input_ids(parts: list[ConversationItem]) -> list[torch.Tensor]:
    """Collect ``type='text'`` token-id tensors (``value``); one tensor per text row."""
    return [part.value for part in parts if part.type == "text"]
