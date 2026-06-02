"""ConversationItem and helpers for SeedOmni V2 inference.

Inference-only data structure (training still rides :class:`TrainConversation`
on the existing ``forward`` / ``pre_forward`` paths — see
:mod:`veomni.models.seed_omni.module`).  The :class:`OmniInferencer` walks a
flat ``List[ConversationItem]`` from input through the FSM; every module's
``generate`` reads / mutates that list and routes it as a single ``ctx``
slot.

Unified item shape
------------------
Each item is ``{type, value, meta}``:

* ``type``: ``"text"`` | ``"image"`` | ``"token"`` | ``"output"`` | ``"soi"`` | ``"eoi"``
* ``value``: polymorphic payload (raw content or embedded tensor)
* ``meta``: ``role`` (``user`` / ``assistant`` / ``system`` / ``dummy``), plus
  optional ``source``, ``input_ids``, ``token_id``, ``phase``, etc.

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
Role = str  # "system" | "user" | "assistant" | "dummy"
ArPhase = Literal["text", "vq"]


@dataclass
class ConversationItem:
    """One element of a conversation list — unified ``{type, value, meta}``."""

    type: ItemType
    value: Any = None
    meta: dict = field(default_factory=dict)


# Backward-compatible alias used across the codebase and tests.
ConversationPart = ConversationItem


def item_role(item: ConversationItem) -> str:
    """Return ``meta["role"]``, defaulting to ``"user"``."""
    return str(item.meta.get("role", "user"))


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


def append_output_hidden(
    parts: list[ConversationItem],
    hidden: torch.Tensor,
    *,
    phase: ArPhase,
    role: str = "assistant",
) -> ConversationItem:
    """Append a new ``output`` item carrying backbone hidden states."""
    item = ConversationItem(
        type="output",
        value=hidden,
        meta={"role": role, "phase": phase},
    )
    parts.append(item)
    return item


def maybe_merge_outputs(parts: list[ConversationItem], *, phase: ArPhase) -> bool:
    """Concatenate the last two ``output`` items when they share ``phase``.

    Returns ``True`` when a merge occurred.
    """
    if len(parts) < 2:
        return False
    a, b = parts[-2], parts[-1]
    if a.type != "output" or b.type != "output":
        return False
    if item_phase(a) != phase or item_phase(b) != phase:
        return False
    emb_a = get_llm_embed(a)
    emb_b = get_llm_embed(b)
    if emb_a is None or emb_b is None:
        return False
    a.value = torch.cat([emb_a, emb_b], dim=1)
    parts.pop()
    return True


def seal_phase_outputs(parts: list[ConversationItem], *, phase: ArPhase, new_type: ItemType) -> int:
    """Rename every ``output`` item with ``meta.phase == phase`` to ``new_type``.

    Returns the number of items renamed.
    """
    count = 0
    for part in parts:
        if part.type != "output":
            continue
        if item_phase(part) != phase:
            continue
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
        parts.append(ConversationItem(type="image", value=img, meta={"role": "user"}))
    parts.append(ConversationItem(type="text", value=prompt, meta={"role": "user"}))
    parts.append(ConversationItem(type="text", value="", meta={"role": "assistant"}))
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
    """Token ids from the text decoder cache mirrored on ``token`` parts.

    Legacy path: reads ``type="token"`` assistant parts (excluding VQ-sourced).
    Prefer :func:`merge_token_cache_into_parts` when using module-private caches.
    """
    out: list[int] = []
    for part in parts:
        if part.type != "token" or item_role(part) != "assistant":
            continue
        if part.meta.get("source") == "vqvae":
            continue
        tid = get_token_id(part)
        if tid is not None:
            out.append(tid)
    return out


# ── Raw training conversation helpers (D3/D4/D5) ────────────────────────────


def is_raw_training_conversation(conversation: Any) -> bool:
    """True iff ``conversation`` is the raw training shape ``list[list[dict]]``."""
    return (
        isinstance(conversation, list)
        and len(conversation) > 0
        and isinstance(conversation[0], list)
        and len(conversation[0]) > 0
        and isinstance(conversation[0][0], dict)
    )


def collect_modality_values(
    conversation_list: list[list[dict]],
    types: tuple[str, ...],
) -> list[list[Any]]:
    """Per-sample list of ``item['value']`` for items whose ``type`` matches."""
    out: list[list[Any]] = []
    for sample in conversation_list:
        out.append([item.get("value") for item in sample if item.get("type") in types])
    return out


# ── Training: the embedding-segment conversation (D5 backbone splice) ────────


@dataclass
class TrainSegment:
    """One contiguous run of token positions with a shared embedding source."""

    embeds: torch.Tensor  # (L, D)
    label_ids: torch.Tensor  # (L,) long — text CE target (-100 = ignore)
    gen_ids: torch.Tensor  # (L,) long — VQ CE target  (-100 = ignore)


@dataclass
class TrainConversation:
    """Single training carrier — the only object that enters the backbone."""

    raw: list[list[dict[str, Any]]]
    und_embeds: torch.Tensor | None = None
    gen_embeds: torch.Tensor | None = None
    gen_token_ids: torch.Tensor | None = None
    segments: list[list[TrainSegment]] | None = None
    hidden_states: torch.Tensor | None = None


def is_train_conversation(x: Any) -> bool:
    """True iff ``x`` is the V2 training carrier :class:`TrainConversation`."""
    return isinstance(x, TrainConversation)


def assemble_embeds(
    segments: list[list[TrainSegment]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-sample segment embeds → right-padded ``(B, T, D)`` batch."""
    if not segments:
        raise ValueError("assemble_embeds: empty segments list.")
    per_sample = [torch.cat([seg.embeds for seg in sample], dim=0) if sample else None for sample in segments]
    if any(x is None for x in per_sample):
        raise ValueError("assemble_embeds: a sample has zero segments — text encoder must emit at least bos+eos.")
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


def assemble_labels(segments: list[list[TrainSegment]], *, key: str) -> torch.Tensor:
    """Concatenate the per-segment ``key`` label vectors → right-padded ``(B, T)``."""
    per_sample = [torch.cat([getattr(seg, key) for seg in sample], dim=0) if sample else None for sample in segments]
    if any(x is None for x in per_sample):
        raise ValueError("assemble_labels: a sample has zero segments.")
    lengths = [x.size(0) for x in per_sample]
    max_len = max(lengths)
    device = per_sample[0].device
    labels = torch.full((len(per_sample), max_len), -100, dtype=torch.long, device=device)
    for i, (lab, n) in enumerate(zip(per_sample, lengths)):
        labels[i, :n] = lab.to(torch.long)
    return labels


__all__ = [
    "ArPhase",
    "ConversationItem",
    "ConversationPart",
    "ItemType",
    "Role",
    "append_output_hidden",
    "build_conversation",
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
    "is_raw_training_conversation",
    "collect_modality_values",
    "TrainSegment",
    "TrainConversation",
    "is_train_conversation",
    "assemble_embeds",
    "assemble_labels",
]
