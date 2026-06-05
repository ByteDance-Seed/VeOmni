"""ConversationItem and helpers for SeedOmni V2 (training + inference).

Both training and inference walk a flat ``List[ConversationItem]`` (or a batch
``list[list[ConversationItem]]``).  Modules read / mutate ``value`` in place
(embeds, hidden states) and route the list as a single ``ctx`` / batch slot.

Unified item shape
------------------
Each item is ``{type, value, role, meta}``:

* ``type``: ``"text"`` | ``"image"`` | ``"token"`` | ``"output"`` | ``"soi"`` | ``"eoi"``
* ``value``: polymorphic payload (raw content or embedded tensor)
* ``role``: ``"user"`` | ``"assistant"`` | ``"dummy"``
* ``meta``: opaque per-module baggage written during forward (``input_ids``, ``phase``, …)

Lifecycle
---------
A part is born with raw ``value`` (``str``, PIL image, token id) and becomes
"embedded" once an encoder overwrites ``value`` with an ``(L, D)`` or
``(1, L, D)`` tensor.

AR workspace
------------
During auto-regressive decoding the backbone appends ``type="output"`` items
carrying hidden states or embeds.  Modality heads (text / VQ) decode the
hidden, replace ``value`` with an embed, and merge adjacent ``output`` items
**within the same** ``meta["phase"]`` only (``"text"`` or ``"vq"``).

Phase boundaries are sealed by renaming completed ``output`` spans to
``type="text"`` or ``type="image"`` so CFG can drop text vs image spans
independently.  ``<begin_of_image>`` / ``<end_of_image>`` are ``soi`` / ``eoi``
items — never merged into text output history.

Sampled token ids live in module-private caches, not in the conversation list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import torch


ItemType = str  # "text" | "image" | "token" | "output" | "soi" | "eoi"
Role = str  # "user" | "assistant" | "dummy"
ArPhase = Literal["text", "vq"]


@dataclass
class ConversationItem:
    """One element of a conversation list — ``{type, value, role, meta}``."""

    type: ItemType
    value: Any = None
    role: Role = "user"
    meta: dict = field(default_factory=dict)


# Backward-compatible alias used across the codebase and tests.
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


def set_llm_embed(item: ConversationItem, embed: torch.Tensor, *, token_id: int | None = None) -> None:
    """Write an embed into ``value``, preserving ``token_id`` in ``meta`` when given."""
    if token_id is not None:
        item.meta["token_id"] = int(token_id)
    elif item.type == "token":
        existing = get_token_id(item)
        if existing is not None:
            item.meta["token_id"] = existing
    item.value = embed


def item_phase(item: ConversationItem) -> str | None:
    """Return ``meta["phase"]`` for ``output`` items, else ``None``."""
    phase = item.meta.get("phase")
    return str(phase) if phase is not None else None


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


def get_ar_tail_embed(parts: list[ConversationItem]) -> torch.Tensor | None:
    """Last-position embed from the conversation tail for one AR step."""
    if not parts:
        return None
    emb = get_llm_embed(parts[-1])
    if emb is None:
        return None
    return emb[:, -1:, :]


def drop_trailing_lm_hidden(parts: list[ConversationItem]) -> None:
    """Drop the last single-token assistant row before a boundary emit (SOI/EOI).

    After ``janus_llama`` runs, the tail is often a one-position embed that must
    not be fed into the backbone again when opening/closing an image span.
    """
    if not parts:
        return
    tail = parts[-1]
    if tail.type == "output":
        parts.pop()
        return
    if tail.role != "assistant" or not tail.meta.get("generated"):
        return
    emb = get_llm_embed(tail)
    if emb is not None and emb.size(1) == 1:
        parts.pop()


def maybe_merge_outputs(parts: list[ConversationItem], *, phase: ArPhase) -> bool:
    """Merge the last two ``output`` rows in the same AR phase (concat on seq dim)."""
    if len(parts) < 2:
        return False
    a, b = parts[-2], parts[-1]
    if a.type != "output" or b.type != "output":
        return False
    if item_phase(a) != phase or item_phase(b) != phase:
        return False
    emb_a, emb_b = get_llm_embed(a), get_llm_embed(b)
    if emb_a is None or emb_b is None:
        return False
    a.value = torch.cat([emb_a, emb_b], dim=1)
    ids_a, ids_b = a.meta.get("input_ids"), b.meta.get("input_ids")
    if isinstance(ids_a, torch.Tensor) and isinstance(ids_b, torch.Tensor):
        a.meta["input_ids"] = torch.cat([ids_a, ids_b], dim=1)
    parts.pop()
    return True


def seal_phase_outputs(parts: list[ConversationItem], *, phase: ArPhase, new_type: ItemType) -> int:
    """Rename completed ``output`` spans to a sealed type (``text`` / ``image``)."""
    count = 0
    for part in parts:
        if part.type == "output" and item_phase(part) == phase:
            part.type = new_type
            part.meta.pop("phase", None)
            count += 1
    return count


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


def iter_kind(parts: list[ConversationItem], kind: ItemType) -> Iterator[ConversationItem]:
    """Alias for :func:`iter_type` (legacy name)."""
    yield from iter_type(parts, kind)


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


def is_batched_conversation_list(conversation: Any) -> bool:
    """True iff ``conversation`` is ``list[list[ConversationItem]]``."""
    return isinstance(conversation, list) and len(conversation) > 0 and isinstance(conversation[0], list)


def collect_modality_values(
    conversation_list: list[list[ConversationItem]],
    types: list[str],
    roles: list[str] | None = None,
) -> list[list[Any]]:
    """Per-sample list of ``item.value`` for items whose ``type`` (and optional ``role``) match."""
    out: list[list[Any]] = []
    for sample in conversation_list:
        row: list[Any] = []
        for item in sample:
            if item.type not in types:
                continue
            if roles is not None and item_role(item) not in roles:
                continue
            row.append(item.value)
        out.append(row)
    return out


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


def sample_indices_with_modality(
    conversation_list: list[list[ConversationItem]],
    types: list[str],
    roles: list[str] | None = None,
) -> set[int]:
    """Micro-batch row indices that contain at least one matching item."""
    found: set[int] = set()
    for sample_idx, sample in enumerate(conversation_list):
        for item in sample:
            if item.type not in types:
                continue
            if roles is not None and item_role(item) not in roles:
                continue
            found.add(sample_idx)
            break
    return found


def collect_modality_batch(
    conversation_list: list[list[ConversationItem]],
    types: list[str],
    roles: list[str] | None = None,
) -> list[Any]:
    """Flat ``item.value`` list for matching items in micro-batch order."""
    return [item.value for item in iter_modality_items(conversation_list, types, roles)]


def iter_embed_chunks(sample: list[ConversationItem]) -> list[torch.Tensor]:
    """Concat-ready ``(L, D)`` embed chunks for one sample (training backbone splice)."""
    chunks: list[torch.Tensor] = []
    for part in sample:
        emb = get_llm_embed(part)
        if emb is None and isinstance(part.value, torch.Tensor) and part.value.dim() == 2:
            emb = part.value.unsqueeze(0)
        if emb is None:
            continue
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        chunks.append(emb)
    return chunks


def assemble_batch_embeds(
    conversations: list[list[ConversationItem]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad per-sample embed chains → ``(inputs_embeds, attention_mask, position_ids)``."""
    if not conversations:
        raise ValueError("assemble_batch_embeds: empty batch.")
    per_sample = [torch.cat(iter_embed_chunks(sample), dim=0) if sample else None for sample in conversations]
    if any(x is None for x in per_sample):
        raise ValueError("assemble_batch_embeds: a sample has zero embedded items.")
    lengths = [x.size(0) for x in per_sample]
    max_len = max(lengths)
    ref = per_sample[0]
    dim = ref.size(-1)
    device = ref.device
    inputs_embeds = ref.new_zeros((len(per_sample), max_len, dim))
    attention_mask = torch.zeros((len(per_sample), max_len), dtype=torch.long, device=device)
    for i, (emb, n) in enumerate(zip(per_sample, lengths)):
        inputs_embeds[i, :n] = emb
        attention_mask[i, :n] = 1
    position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(len(per_sample), -1)
    return inputs_embeds, attention_mask, position_ids


def assemble_batch_labels(
    conversations: list[list[ConversationItem]],
    *,
    key: str,
) -> torch.Tensor:
    """Concat per-item ``meta[key]`` vectors → right-padded ``(B, T)``."""
    per_sample: list[torch.Tensor | None] = []
    for sample in conversations:
        parts: list[torch.Tensor] = []
        for part in sample:
            lab = part.meta.get(key)
            if lab is None:
                continue
            if not isinstance(lab, torch.Tensor):
                raise TypeError(f"ConversationItem.meta[{key!r}] must be a tensor, got {type(lab).__name__}.")
            parts.append(lab.to(torch.long))
        per_sample.append(torch.cat(parts, dim=0) if parts else None)
    if any(x is None for x in per_sample):
        raise ValueError(f"assemble_batch_labels: a sample has no meta[{key!r}] segments.")
    lengths = [x.size(0) for x in per_sample]
    max_len = max(lengths)
    device = per_sample[0].device
    labels = torch.full((len(per_sample), max_len), -100, dtype=torch.long, device=device)
    for i, (lab, n) in enumerate(zip(per_sample, lengths)):
        labels[i, :n] = lab
    return labels


def assemble_batch_gen_image_mask(
    conversations: list[list[ConversationItem]],
) -> torch.Tensor:
    """``(B, T)`` bool mask — True wherever ``meta.gen_ids`` is supervised."""
    gen_labels = assemble_batch_labels(conversations, key="gen_ids")
    return gen_labels.ne(-100)


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


def dummy_modality_anchor_mean(
    conversations: list[list[ConversationItem]],
    *,
    item_type: str,
) -> torch.Tensor | None:
    """Mean of ``role='dummy'`` rows for one modality — QwenVL-style FSDP anchor."""
    total: torch.Tensor | None = None
    count = 0
    for sample in conversations:
        for part in sample:
            if part.type != item_type or not is_dummy(part):
                continue
            if not isinstance(part.value, torch.Tensor):
                continue
            total = part.value.mean() if total is None else total + part.value.mean()
            count += 1
    if total is None or count == 0:
        return None
    return total / count


def modality_embed_anchor_mean(
    conversations: list[list[ConversationItem]],
    *,
    item_type: str,
) -> torch.Tensor | None:
    """Mean of every ``item_type`` embed on the carrier (real + dummy rows)."""
    total: torch.Tensor | None = None
    count = 0
    for sample in conversations:
        for part in sample:
            if part.type != item_type or not isinstance(part.value, torch.Tensor):
                continue
            total = part.value.mean() if total is None else total + part.value.mean()
            count += 1
    if total is None or count == 0:
        return None
    return total / count


__all__ = [
    "ArPhase",
    "ConversationItem",
    "ConversationPart",
    "ItemType",
    "Role",
    "build_conversation",
    "drop_trailing_lm_hidden",
    "collect_prompt_embeds",
    "get_ar_tail_embed",
    "item_phase",
    "item_role",
    "is_dummy",
    "is_embedded",
    "maybe_merge_outputs",
    "needs_embedding",
    "get_token_id",
    "get_llm_embed",
    "set_llm_embed",
    "seal_phase_outputs",
    "iter_type",
    "iter_kind",
    "unembedded_parts",
    "latest_assistant_text_token_ids",
    "is_batched_conversation_list",
    "collect_modality_values",
    "iter_modality_items",
    "sample_indices_with_modality",
    "collect_modality_batch",
    "iter_embed_chunks",
    "assemble_batch_embeds",
    "assemble_batch_labels",
    "assemble_batch_gen_image_mask",
    "summarize_conversation_batch",
    "dummy_modality_anchor_mean",
    "modality_embed_anchor_mean",
]
