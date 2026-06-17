"""Qwen3-VL ChatML template (text + image + video) as readable Python.

Mirrors the upstream ``chat_template.json``:

* each turn is wrapped in ``<|im_start|>{role}\\n … <|im_end|>\\n``;
* image / video become ``<|vision_start|><|image_pad|><|vision_end|>`` /
  ``<|vision_start|><|video_pad|><|vision_end|>`` — in the V2 segment model the
  ``<|*_pad|>`` run is *not* tokenized; the sibling ``image`` / ``video`` item
  already carries the merged vision tokens, so the template emits
  ``<|vision_start|>`` text · the media item · ``<|vision_end|>`` text.

Qwen3-VL has no audio modality (audio-in-video is an Omni feature — see
``design.md`` § av-video, design-only).

Training flow: :func:`apply_qwen3vl_chat_template` → tokenize text rows → merge
adjacent text → :func:`pack_text_input_ids`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ....conversation import ConversationItem


@dataclass(frozen=True)
class Qwen3VLChatMarkers:
    """Wire-format strings resolved from the module tokenizer."""

    im_start_token: str
    im_end_token: str
    eos_token: str
    assistant_prefix: str
    vision_start_token: str
    vision_end_token: str


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


def apply_qwen3vl_chat_template(
    sample: list[ConversationItem],
    markers: Qwen3VLChatMarkers,
) -> list[ConversationItem]:
    """Apply Qwen3-VL ChatML to a raw conversation (text + image parts)."""
    out: list[ConversationItem] = []
    dummy_parts: list[ConversationItem] = []
    prev_role: str | None = None

    def close_turn(role: str) -> None:
        out.append(_template_item("text", markers.im_end_token + "\n", role, loss_mask=int(role == "assistant")))

    for item in sample:
        role = item.role
        if role == "dummy":
            dummy_parts.append(item)
            continue

        if role != prev_role:
            if prev_role is not None:
                close_turn(prev_role)
            out.append(_template_item("text", markers.im_start_token + role + "\n", role, loss_mask=0))
            prev_role = role

        if item.type == "text":
            out.append(_template_item("text", str(item.value), role))
        elif item.type in ("image", "video"):
            # Image and video both wrap in <|vision_start|> … <|vision_end|>
            # (the model uses <|image_pad|> / <|video_pad|> inside). Qwen3-VL has
            # no audio modality — audio-in-video is an Omni feature (design-only,
            # see design.md § av-video).
            out.append(_template_item("text", markers.vision_start_token, role, loss_mask=0))
            out.append(_template_item(item.type, item.value, role, meta=dict(item.meta)))
            out.append(_template_item("text", markers.vision_end_token, role, loss_mask=0))
        else:
            raise ValueError(f"Qwen3-VL text encoder only supports text/image/video items, got {item.type!r}")

    if prev_role is not None:
        close_turn(prev_role)
    out.extend(dummy_parts)
    return out


def apply_qwen3vl_generation_prompt(
    sample: list[ConversationItem],
    markers: Qwen3VLChatMarkers,
) -> list[ConversationItem]:
    """Append the assistant generation prefix after a templated (turn-closed) prompt."""
    out = list(sample)
    out.append(_template_item("text", markers.assistant_prefix, "assistant", loss_mask=0))
    return out


def pack_text_input_ids(parts: list[ConversationItem]) -> list[torch.Tensor]:
    """Collect ``type='text'`` token-id tensors (``value``); one tensor per text row."""
    return [part.value for part in parts if part.type == "text"]
