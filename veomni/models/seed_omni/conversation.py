"""``ConversationItem`` — the single carrier object of SeedOmni V2.

The whole pipeline (training and inference) operates on one batched
``conversation_list`` (``list[list[ConversationItem]]`` for training, a flat
``list[ConversationItem]`` for one inference request).  Modules read and
**mutate ``item.value`` in place** — there are no per-field edge channels;
``OmniModel`` writes the (possibly grown) list back into the shared batch /
``ctx`` after every node.

Item shape
----------
Each item is ``{type, value, role, meta}``:

* ``type``  — ``"text"`` | ``"image"`` | ``"output"`` (and the legacy
  ``"token"`` shape consumed by :func:`get_token_id`).
* ``value`` — polymorphic: raw content (``str`` / PIL image / pixel tensor)
  before encoding, an ``(L, D)`` / ``(1, L, D)`` embedding tensor after.
* ``role``  — ``"user"`` | ``"assistant"`` | ``"dummy"`` (``"dummy"`` rows are
  zero-tensor FSDP placeholders appended by encoders on text-only
  micro-batches; the backbone skips them and folds a zero-grad anchor).
* ``meta``  — per-module baggage written during forward (``labels`` /
  ``attention_mask`` / ``janus_vqvae_labels`` / ``source`` / …).

Lifecycle
---------
1. An item is born with a raw ``value`` (text string, PIL image, pixels).
2. An encoder (SigLIP / VQVAE / text wte) overwrites ``value`` with its
   embedding tensor.
3. The backbone overwrites ``value`` again with the per-segment hidden state.
4. During inference, backbone steps append ``type="output"`` rows; when a
   modality span finishes, :func:`seal_outputs` renames the trailing
   ``output`` row to ``"text"`` / ``"image"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


ItemType = str  # "text" | "image" | "output" | "token"
Role = str  # "user" | "assistant" | "dummy"


@dataclass
class ConversationItem:
    """One element of a conversation list — ``{type, value, role, meta}``."""

    type: ItemType
    value: Any = None
    role: Role = "user"
    meta: dict = field(default_factory=dict)


# Public alias kept for callers that import the older name.
ConversationPart = ConversationItem


def item_role(item: ConversationItem) -> str:
    """Return ``item.role``."""
    return str(item.role)


def is_dummy(item: ConversationItem) -> bool:
    return item_role(item) == "dummy"


def _is_chw_image_tensor(value: torch.Tensor) -> bool:
    """True for raw ``(C, H, W)`` pixels — not ``(1, P, D)`` SigLIP patch embeds."""
    if value.dim() != 3:
        return False
    c, _h, w = (int(value.size(0)), int(value.size(1)), int(value.size(2)))
    if c not in (1, 3):
        return False
    # SigLIP writes ``(1, num_patches, hidden)`` back onto the item — the last
    # axis is the LLM hidden size (≫ any spatial extent), not image width.
    if c == 1 and w > 512:
        return False
    return True


def is_embedded(item: ConversationItem) -> bool:
    """True when ``value`` holds an LLM-space embedding tensor."""
    value = item.value
    if not isinstance(value, torch.Tensor):
        return False
    if item.type in ("text", "output", "soi", "eoi"):
        return value.dim() == 3
    if item.type == "image":
        if _is_chw_image_tensor(value):
            return False
        return value.dim() >= 2
    if item.type == "token":
        if isinstance(value, int):
            return False
        if value.numel() == 1:
            return False
        return value.dim() >= 2
    return False


def needs_embedding(item: ConversationItem) -> bool:
    """True when an encoder still needs to fill ``value`` with an embed."""
    if is_dummy(item) or is_embedded(item):
        return False
    if item.type == "text":
        return bool(item.value)
    if item.type == "token":
        tid = get_token_id(item)
        return tid is not None
    return False


def get_token_id(item: ConversationItem) -> int | None:
    """Extract a scalar token id from a ``type="token"`` item."""
    if item.type != "token":
        return None
    if "token_id" in item.meta:
        return int(item.meta["token_id"])
    value = item.value
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.reshape(-1)[0].item())
    return None


def get_llm_embed(item: ConversationItem) -> torch.Tensor | None:
    """Return ``(B, T, D)`` embed for LLM consumption, or ``None``."""
    if not is_embedded(item):
        return None
    value = item.value
    assert isinstance(value, torch.Tensor)
    if item.type == "image":
        if value.dim() == 2:
            return value.unsqueeze(0)
        if value.dim() == 3:
            return value
    if item.type in ("text", "output", "soi", "eoi") and value.dim() == 3:
        return value
    if item.type == "token":
        if value.dim() == 3:
            return value
        if value.dim() == 2:
            return value.unsqueeze(1)
    return None


def collect_prompt_embeds(parts: list[ConversationItem]) -> list[torch.Tensor]:
    """Embedded history for the backbone prompt pass — excludes active ``output`` items."""
    chunks: list[torch.Tensor] = []
    for part in parts:
        if is_dummy(part) or part.type == "output":
            continue
        emb = get_llm_embed(part)
        if emb is not None:
            chunks.append(emb)
    return chunks


def maybe_merge_outputs(parts: list[ConversationItem]) -> bool:
    """Merge the last two ``output`` rows in the same AR phase (concat on seq dim)."""
    if len(parts) < 2:
        return False
    a, b = parts[-2], parts[-1]
    if a.type != "output" or b.type != "output":
        return False
    emb_a, emb_b = a.value, b.value
    emb_b = emb_b.to(device=emb_a.device, dtype=emb_a.dtype)
    a.value = torch.cat([emb_a, emb_b], dim=0)
    parts.pop()
    return True


def seal_outputs(parts: list[ConversationItem], new_type: ItemType) -> int:
    """Rename completed ``output`` spans to a sealed type (``text`` / ``image``)."""
    assert parts[-1].type == "output"
    parts[-1].type = new_type


def build_conversation(
    *,
    prompt: str,
    images: list[Any] | None = None,
) -> list[ConversationItem]:
    """Build the canonical conversation list for a single inference request."""
    parts: list[ConversationItem] = []
    for img in images or []:
        parts.append(ConversationItem(type="image", value=img, role="user"))
    parts.append(ConversationItem(type="text", value=prompt, role="user"))
    return parts


def iter_type(parts: list[ConversationItem], item_type: ItemType) -> Iterator[ConversationItem]:
    """Yield parts of a given ``type`` in declaration order."""
    for part in parts:
        if part.type == item_type:
            yield part


def unembedded_parts(parts: list[ConversationItem]) -> list[ConversationItem]:
    """Parts that still need an encoder pass."""
    return [p for p in parts if needs_embedding(p)]


def latest_assistant_text_token_ids(parts: list[ConversationItem]) -> list[int]:
    """Token ids from generated assistant ``text`` rows (``meta.generated=True``)."""
    out: list[int] = []
    for part in parts:
        if part.type != "text" or item_role(part) != "assistant":
            continue
        if not part.meta.get("generated"):
            continue
        if part.meta.get("vq_token"):
            continue
        tid = part.meta.get("token_id")
        if tid is not None:
            out.append(int(tid))
            continue
        ids = part.meta.get("input_ids")
        if isinstance(ids, torch.Tensor) and ids.numel() == 1:
            out.append(int(ids.reshape(-1)[0].item()))
    return out


# ── Training batch helpers (unified with inference ConversationItem) ────────


def iter_modality_items(
    conversation_list: list[list[ConversationItem]],
    types: list[str],
    roles: list[str] | None = None,
) -> Iterator[ConversationItem]:
    """Yield matching items in micro-batch order (sample 0, then sample 1, …)."""
    for sample in conversation_list:
        for item in sample:
            if item.type not in types:
                continue
            if roles is not None and item_role(item) not in roles:
                continue
            yield item


def collect_modality_batch(
    conversation_list: list[list[ConversationItem]],
    types: list[str],
    roles: list[str] | None = None,
) -> list[Any]:
    """Flat ``item.value`` list for matching items in micro-batch order."""
    return [item.value for item in iter_modality_items(conversation_list, types, roles)]


def _value_shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, str):
        preview = value[:40] + "..." if len(value) > 40 else value
        return f"str({len(value)}):{preview!r}"
    if value is None:
        return "None"
    return type(value).__name__


def summarize_conversation_batch(conversations: Any) -> str:
    """One-line-per-item debug view: ``sample[i] type shape role=… dummy=…``."""
    if not isinstance(conversations, list) or not conversations:
        return "(empty conversation_list)"
    lines: list[str] = []
    for si, sample in enumerate(conversations):
        if not isinstance(sample, list):
            lines.append(f"  sample[{si}]: {type(sample).__name__}")
            continue
        lines.append(f"  sample[{si}] ({len(sample)} items):")
        for j, item in enumerate(sample):
            typ = str(item.type)
            role = item_role(item)
            dummy = is_dummy(item)
            val = item.value
            lines.append(f"    [{j}] type={typ!r} shape={_value_shape(val)} role={role!r} dummy={dummy}")
    return "\n".join(lines)


__all__ = [
    "ConversationItem",
    "ConversationPart",
    "ItemType",
    "Role",
    "build_conversation",
    "collect_prompt_embeds",
    "item_role",
    "is_dummy",
    "is_embedded",
    "maybe_merge_outputs",
    "needs_embedding",
    "get_token_id",
    "get_llm_embed",
    "seal_outputs",
    "iter_type",
    "unembedded_parts",
    "latest_assistant_text_token_ids",
    "iter_modality_items",
    "collect_modality_batch",
    "summarize_conversation_batch",
]
