"""Janus chat template as readable Python (mirrors ``chat_template.jinja``).

Training flow
-------------
1. :func:`apply_janus_chat_template` — insert bos / system / role markers /
   boi–image–eoi spans; set ``meta["loss_mask"]`` on template rows that differ
   from the default ``int(role == "assistant")``.
2. :func:`render_template_string` — concatenate the human-readable wire string.
3. Text encoder tokenizes (token ids in ``value``), merges adjacent text, then packs text rows.
4. :func:`pack_text_input_ids` — ``list[Tensor]`` of per-text-row token ids (for ``naflatten``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ....conversation import ConversationItem


# Upstream Jinja emits this literal for ``content['type'] == 'image'``.
IMAGE_PLACEHOLDER = "<image_placeholder>"


@dataclass(frozen=True)
class JanusChatMarkers:
    """Wire-format strings — use ``tokenizer.bos_token``, ``tokenizer.boi_token``, etc."""

    bos_token: str
    eos_token: str
    boi_token: str
    eoi_token: str
    system_prompt: str
    user_prefix: str
    assistant_prefix: str


def _template_item(
    item_type: str,
    value: Any,
    role: str,
    *,
    loss_mask: int | None = None,
    meta: dict | None = None,
) -> ConversationItem:
    """Build one conversation row; ``text`` rows always get ``meta["loss_mask"]``."""
    part_meta = dict(meta or {})
    if item_type == "text":
        part_meta["loss_mask"] = int(role == "assistant") if loss_mask is None else int(loss_mask)
    return ConversationItem(type=item_type, value=value, role=role, meta=part_meta)


def apply_janus_chat_template(
    sample: list[ConversationItem],
    markers: JanusChatMarkers,
) -> list[ConversationItem]:
    """Apply Janus chat template to a raw conversation (text + image parts)."""
    out: list[ConversationItem] = []
    dummy_parts: list[ConversationItem] = []  # dummy from other modules
    out.append(_template_item("text", markers.bos_token, "system"))
    out.append(_template_item("text", markers.system_prompt, "system"))
    prev_role: str | None = None
    for item in sample:
        role = item.role
        if role != prev_role:
            if role == "user":
                out.append(_template_item("text", markers.user_prefix, "user"))
            elif role == "assistant":
                out.append(_template_item("text", markers.assistant_prefix, "assistant", loss_mask=0))
            prev_role = role

        if item.type == "text":
            out.append(_template_item("text", item.value, role))
        elif item.type == "image" and role != "dummy":
            out.append(_template_item("text", markers.boi_token, role))
            out.append(_template_item("image", item.value, role, meta=dict(item.meta)))
            out.append(_template_item("text", markers.eoi_token, role))
        elif role == "dummy":
            dummy_parts.append(item)
        else:
            out.append(_template_item(item.type, item.value, role, meta=dict(item.meta)))

    out.append(_template_item("text", markers.eos_token, "assistant"))
    out.extend(dummy_parts)
    return out


def render_template_string(parts: list[ConversationItem]) -> str:
    """Build the on-the-wire prompt string (Jinja-visible layout)."""
    chunks: list[str] = []
    for part in parts:
        if part.type == "text":
            chunks.append(str(part.value or ""))
        elif part.type == "image":
            chunks.append(IMAGE_PLACEHOLDER)
        else:
            raise ValueError(f"Unsupported part type: {part.type}")
    return "".join(chunks)


def pack_text_input_ids(parts: list[ConversationItem]) -> list[torch.Tensor]:
    """Collect ``type='text'`` token-id tensors (``value``); one tensor per text row."""
    return [part.value for part in parts if part.type == "text"]
