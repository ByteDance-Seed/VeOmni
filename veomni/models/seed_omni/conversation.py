"""ConversationPart and helpers for SeedOmni V2 inference.

Inference-only data structure (training still rides input_ids + masked_scatter
on the existing ``forward`` / ``pre_forward`` paths — see
:mod:`veomni.models.seed_omni.module`).  The :class:`OmniInferencer` walks a
flat ``List[ConversationPart]`` from input through the FSM; every module's
``generate`` reads / mutates that list and routes it as a single ``ctx``
slot.  No chat template is used: the text encoder owns bos / boi / eoi
placement and decides the on-the-wire layout.

Lifecycle of a part
-------------------
A part is born holding raw content (``text``, ``image``, ``token_id``) and
becomes "encoded" once a module fills in ``inputs_embeds``.  The backbone
LLM concatenates every part's ``inputs_embeds`` on the prompt pass and only
the latest part's ``inputs_embeds`` on each AR step (KV cache hot path).

Layout invariants enforced by :func:`build_conversation`
--------------------------------------------------------
1. Images come first.  Every user image is appended as an
   ``image_und`` part (``role=user``) before any text part.
2. A single user text part follows (``role=user``).
3. A trailing empty assistant text part is appended as the completion
   marker (``role=assistant``).  ``text_encoder.generate`` will append
   sampled-token parts under this assistant role; emitted boundary
   tokens (``boi`` / ``eoi``) and VQ tokens piggy-back on the same
   role.

These invariants mean the conversation list is monotonically growing
during a generate call — no part is ever removed.  ``token``-kind parts
that follow the trailing assistant marker form the assistant's response.

Sampled-token parts
-------------------
``kind="token"`` parts are produced by:

* :meth:`JanusTextEncoder.decode`        — one part per sampled text token,
                                            also for the t2i-forced boi.
* :meth:`JanusTextEncoder.emit_image_*`  — one part per boundary token.
* :meth:`JanusVqvae.generate`            — one part per sampled VQ token.

Each carries ``inputs_embeds`` pre-filled by the producing module so the
next FSM step can re-use it through ``janus_llama.generate``'s "tail-only"
fast path without re-embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


PartKind = str  # "text" | "image_und" | "image_gen" | "token"
Role = str  # "system" | "user" | "assistant"


@dataclass
class ConversationPart:
    """One element of a conversation list.

    The dataclass is intentionally a flat record (no nested Union types) so
    a quick check on ``part.kind`` is the source of truth for which fields
    are populated.  All inputs / outputs of every inference module are
    expressed in terms of this single shape.
    """

    kind: PartKind
    role: Role
    # ── Raw content (one of these is set at construction) ────────────────────
    text: str | None = None
    image: Any | None = None  # PIL.Image, numpy array, or pre-computed pixel tensor
    pixel_values: torch.Tensor | None = None  # filled by the inference processor
    token_id: int | None = None
    # ── Encoded form (filled by encoder modules at runtime) ──────────────────
    input_ids: torch.Tensor | None = None  # (1, T) for text parts after tokenisation
    inputs_embeds: torch.Tensor | None = None  # (1, T, hidden_size) after embedding
    # ── Free-form metadata (e.g. processor flags, generated image tensor) ────
    meta: dict = field(default_factory=dict)


def build_conversation(
    *,
    prompt: str,
    images: list[Any] | None = None,
    force_image_gen: bool = False,
) -> list[ConversationPart]:
    """Build the canonical conversation list for a single inference request.

    Layout (fixed):

    * ``image_und`` parts for every user image (in order).
    * One ``text`` user part holding the full prompt.
    * One empty ``text`` assistant part as the completion marker.

    ``force_image_gen`` is consumed by :class:`OmniInferencer` at the
    ``request`` level and does not change the conversation layout.  The
    parameter is accepted here purely for caller convenience so callers can
    forward a single kwargs dict.
    """
    del force_image_gen  # handled upstream via request["force_image_gen"]
    parts: list[ConversationPart] = []
    for img in images or []:
        parts.append(ConversationPart(kind="image_und", role="user", image=img))
    parts.append(ConversationPart(kind="text", role="user", text=prompt))
    parts.append(ConversationPart(kind="text", role="assistant", text=""))
    return parts


def iter_kind(parts: list[ConversationPart], kind: PartKind) -> Iterator[ConversationPart]:
    """Yield parts of a given kind in declaration order."""
    for p in parts:
        if p.kind == kind:
            yield p


def unembedded_parts(parts: list[ConversationPart]) -> list[ConversationPart]:
    """Parts that still need an encoder pass (no ``inputs_embeds`` yet)."""
    return [p for p in parts if p.inputs_embeds is None]


def latest_assistant_text_token_ids(parts: list[ConversationPart]) -> list[int]:
    """Token ids contributed by ``text_encoder.decode`` for the current turn.

    Pulled out so :meth:`JanusTextEncoder.finalize` can detokenize the
    response without re-walking the whole list itself.  Excludes parts
    written by non-text producers (the VQVAE tags its parts with
    ``meta["source"] == "vqvae"``) — VQ codebook ids share the
    ``role="assistant"`` namespace but live in a different vocab and
    must not be fed to the text tokenizer's ``decode``.
    """
    return [
        int(p.token_id)
        for p in parts
        if p.kind == "token" and p.role == "assistant" and p.token_id is not None and p.meta.get("source") != "vqvae"
    ]


__all__ = [
    "ConversationPart",
    "PartKind",
    "Role",
    "build_conversation",
    "iter_kind",
    "unembedded_parts",
    "latest_assistant_text_token_ids",
]
