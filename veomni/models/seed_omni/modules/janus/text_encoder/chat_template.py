"""Janus chat template as readable Python (mirrors ``chat_template.jinja``).

Training flow
-------------
1. :func:`apply_janus_chat_template` — insert bos / (optional) system / role
   markers / boi–image–eoi spans; set ``meta["loss_mask"]`` on template rows
   that differ from the default ``int(role == "assistant")``.
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


def sample_has_user_image(sample: list[ConversationItem]) -> bool:
    """Return True when the raw conversation includes a user ``image`` row."""
    return any(item.type == "image" and item.role == "user" for item in sample)


def apply_janus_chat_template(
    sample: list[ConversationItem],
    markers: JanusChatMarkers,
) -> list[ConversationItem]:
    """Apply Janus chat template to a raw conversation (text + image parts).

    User and assistant images both use explicit ``<boi>`` … ``<eoi>`` text
    rows bracketing a sibling ``image`` row (SigLIP patch embeds).  No
    ``<image_placeholder>`` text row — nothing in the inference graph
    tokenises or embeds that literal.
    """
    out: list[ConversationItem] = []
    dummy_parts: list[ConversationItem] = []  # dummy from other modules
    out.append(_template_item("text", markers.bos_token, "user"))
    # HF Janus ``task != "gen"`` prepends the default system prompt for I2T.
    # TODO: shared by training + inference.  Official Janus I2T prepends the VL
    # system preamble; T2I does not.  Current micro-batches are either pure I2T
    # (user image present) or pure T2I (text-only user turn) — use that as a
    # proxy.  Ideally every sample would carry an explicit system row upstream.
    if sample_has_user_image(sample):
        out.append(_template_item("text", markers.system_prompt, "user"))
    prev_role: str | None = None
    prev_was_user_image = (
        False  # True after a user image; prepend \n to the next user text (HF Jinja same-turn layout).
    )
    for item in sample:
        role = item.role
        if role != prev_role:
            if role == "user":
                out.append(_template_item("text", markers.user_prefix, "user"))
            elif role == "assistant":
                out.append(_template_item("text", markers.assistant_prefix, "assistant", loss_mask=0))
            prev_role = role
            prev_was_user_image = False

        if item.type == "text":
            text = str(item.value)
            if prev_was_user_image and role == "user" and not text.startswith("\n"):
                text = "\n" + text
            out.append(_template_item("text", text, role))
            prev_was_user_image = False
        elif item.type == "image" and role != "dummy":
            out.append(_template_item("text", markers.boi_token, role))
            out.append(_template_item("image", item.value, role, meta=dict(item.meta)))
            out.append(_template_item("text", markers.eoi_token, role))
            prev_was_user_image = role == "user"
        elif role == "dummy":
            dummy_parts.append(item)
        else:
            raise ValueError(f"Unsupported part type: {item.type}")
    if prev_role == "assistant":
        out.append(_template_item("text", markers.eos_token, "assistant"))
    out.extend(dummy_parts)
    return out


def render_template_string(parts: list[ConversationItem]) -> str:  # for debug or demo
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


def tokenize_template_parts(
    parts: list[ConversationItem],
    tokenizer: Any,
    device: Any = None,
) -> None:
    """Tokenize each ``text`` part in place: ``str`` value → token-id tensor.

    Builds tensors on ``device`` (default ``None`` = CPU, used by the worker-side
    preprocessor so no CUDA is touched; the in-module fallback passes the module
    device). Sets ``meta['labels']`` (``-100`` where ``loss_mask`` is 0) and
    ``meta['attention_mask']``.
    """
    for part in parts:
        if part.type == "text":
            text = part.value
            loss_mask = int(part.meta.pop("loss_mask"))
            input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            labels = input_ids if loss_mask else [-100] * len(input_ids)
            part.value = torch.tensor(input_ids, device=device, dtype=torch.long)
            part.meta["labels"] = torch.tensor(labels, device=device, dtype=torch.long)
            part.meta["attention_mask"] = torch.ones(len(input_ids), dtype=torch.long, device=device)


def merge_consecutive_text_parts(parts: list[ConversationItem]) -> list[ConversationItem]:
    """Merge adjacent same-role ``text`` parts (concat ids / labels / mask)."""
    merged: list[ConversationItem] = []
    for part in parts:
        if merged and merged[-1].type == "text" and part.type == "text" and merged[-1].role == part.role:
            prev = merged[-1]
            prev.value = torch.cat([prev.value, part.value])
            prev.meta["labels"] = torch.cat([prev.meta["labels"], part.meta["labels"]])
            prev.meta["attention_mask"] = torch.cat([prev.meta["attention_mask"], part.meta["attention_mask"]])
            continue
        merged.append(part)
    return merged
