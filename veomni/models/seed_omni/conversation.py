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
) -> list[ConversationPart]:
    """Build the canonical conversation list for a single inference request.

    Layout (fixed):

    * ``image_und`` parts for every user image (in order).
    * One ``text`` user part holding the full prompt.
    * One empty ``text`` assistant part as the completion marker.

    Whether the run generates an image is decided by the scenario graph
    (``omni_infer_type`` selects the ``generation_graph``), not by the
    conversation layout — so this helper has no image-generation knob.
    """
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


# ── Raw training conversation helpers (D3/D4/D5) ────────────────────────────
#
# Training feeds modules the *raw* conversation produced by
# ``seedomni_transform`` — ``list[list[dict]]`` where each per-sample list
# holds ``{type, value, role, loss_mask}`` items (``value`` is a string for
# ``text`` and a ``(C, H, W)`` uint8 tensor for ``image`` / ``vq_image``).
# This is distinct from the inference :class:`ConversationPart` objects, so
# modules dispatch their training tokenisation / image extraction on the
# shape below.


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
    """Per-sample list of ``item['value']`` for items whose ``type`` matches.

    Preserves source order so a sample's k-th image lines up with the k-th
    placeholder span the text encoder emits.  Samples without any matching
    modality yield an empty list (the caller fills a zero placeholder so the
    batch dimension — and thus the FSDP graph — stays aligned).
    """
    out: list[list[Any]] = []
    for sample in conversation_list:
        out.append([item.get("value") for item in sample if item.get("type") in types])
    return out


# ── Training: the embedding-segment conversation (D5 backbone splice) ────────
#
# In the V2 target contract (seedomni-v2 SKILL.md, Layer 5/6 + invariants
# 16 & 18) training data does NOT flow as flat ``input_ids`` / ``attention_mask``
# / ``position_ids`` / ``gen_image_mask`` / ``labels`` tensors routed through
# separate edges.  Instead a single object — a :class:`TrainConversation` — is
# the only thing entering the backbone.  Every module fills its own segments'
# ``value`` (= embedding) into it:
#
#   * ``JanusSiglip``  → understanding patch embeds (``und_embeds``)
#   * ``JanusVqvae``   → generation patch embeds + teacher VQ ids
#                        (``gen_embeds`` / ``gen_token_ids``)
#   * ``JanusTextEncoder`` → walks the raw conversation, applies the chat
#                        template, tokenises text + wte-embeds it, wraps images
#                        with ``<boi>``/``<eoi>`` and emits the ordered
#                        per-sample :class:`TrainSegment` list.
#
# The backbone then just concatenates ``segment.embeds`` (segment-order-driven
# splice — NO ``masked_scatter``, NO ``<image_k>`` positional tokens) and
# derives ``attention_mask`` / ``position_ids`` from segment lengths.  Each
# loss head concatenates the matching per-segment label vector
# (``label_ids`` for the text CE, ``gen_ids`` for the VQ CE).  Because the
# backbone and the heads both walk the **same** segment order with the **same**
# right-pad, the assembled ``inputs_embeds`` and the label tensors are aligned
# by construction — no position bookkeeping needed.


@dataclass
class TrainSegment:
    """One contiguous run of token positions with a shared embedding source.

    ``embeds`` is the segment ``value`` (``(L, D)`` — wte embeddings for a text
    run, SigLIP / VQ patch embeddings for an image run, a single wte row for a
    ``<boi>``/``<eoi>`` boundary).  ``label_ids`` / ``gen_ids`` are the
    per-position CE targets for the text head and the VQ head respectively
    (``-100`` where the head should ignore the position).  All three are the
    same length ``L``.
    """

    embeds: torch.Tensor  # (L, D)
    label_ids: torch.Tensor  # (L,) long — text CE target (-100 = ignore)
    gen_ids: torch.Tensor  # (L,) long — VQ CE target  (-100 = ignore)


@dataclass
class TrainConversation:
    """Single training carrier — the only object that enters the backbone.

    Flows ``siglip → vae → text_encoder → janus_llama → {tok_decode,
    vae_decode}`` as one ``conversation_list`` edge.  ``raw`` is kept so the
    vision / VQ / text modules can still extract their modality items.  The
    full-batch ``und_embeds`` / ``gen_embeds`` are retained even after the
    text encoder slices them into segments: the backbone adds a
    ``embeds.sum() * 0.0`` autograd anchor over the **whole** batch tensor so a
    micro-batch with no image of a given modality still drives a (zero)
    gradient through the encoder, keeping FSDP DP grad-reduce aligned
    (seedomni-v2 invariant 10 / the grad-sync anchor).
    """

    raw: list[list[dict[str, Any]]]
    und_embeds: torch.Tensor | None = None  # (B, P, D) understanding patch embeds (also FSDP anchor)
    gen_embeds: torch.Tensor | None = None  # (B, P, D) generation patch embeds (also FSDP anchor)
    gen_token_ids: torch.Tensor | None = None  # (B, P) teacher VQ ids
    segments: list[list[TrainSegment]] | None = None  # per-sample ordered segments (text encoder fills)


def is_train_conversation(x: Any) -> bool:
    """True iff ``x`` is the V2 training carrier :class:`TrainConversation`."""
    return isinstance(x, TrainConversation)


def assemble_embeds(
    segments: list[list[TrainSegment]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-sample segment embeds → right-padded ``(B, T, D)`` batch.

    Returns ``(inputs_embeds, attention_mask, position_ids)`` where
    ``attention_mask`` is ``1`` over real tokens (``0`` on the right pad) and
    ``position_ids`` is ``arange(T)`` per row.  ``T`` is the max assembled
    length across the batch — the *same* ``T`` :func:`assemble_labels` uses, so
    the heads' label tensors line up with these embeds position-for-position.
    """
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
    """Concatenate the per-segment ``key`` label vectors → right-padded ``(B, T)``.

    ``key`` is ``"label_ids"`` (text CE) or ``"gen_ids"`` (VQ CE).  Pads with
    ``-100`` so the right pad is ignored by ``cross_entropy``.  Uses the same
    segment order + ``T`` as :func:`assemble_embeds`.
    """
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
    "ConversationPart",
    "PartKind",
    "Role",
    "build_conversation",
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
