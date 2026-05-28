"""Unit tests for the SeedOmni V2 inference conversation-list helpers.

Scope
-----
Pure-Python tests covering :mod:`veomni.models.seed_omni.conversation`:

* :class:`ConversationPart` is a flat dataclass — sets of fields per
  ``kind`` are by convention, not enforced.  We verify the helpers
  populate them correctly.
* :func:`build_conversation` always lays out parts in the canonical
  order (images → user text → assistant marker) regardless of input
  size, including the empty-image / multi-image edge cases.
* The traversal helpers (``iter_kind``, ``unembedded_parts``,
  ``latest_assistant_text_token_ids``) keep their stable shapes —
  this is what :class:`JanusTextEncoder.finalize` and
  :class:`OmniInferencer` rely on.

No torch / no real modules — this file must import without any GPU /
weights / tokenizer dependency.
"""

from __future__ import annotations

import torch

from veomni.models.seed_omni.conversation import (
    ConversationPart,
    build_conversation,
    iter_kind,
    latest_assistant_text_token_ids,
    unembedded_parts,
)


# ── build_conversation layout invariants ─────────────────────────────────────


def test_build_conversation_text_only_yields_two_parts():
    parts = build_conversation(prompt="hello")
    assert [(p.kind, p.role, p.text) for p in parts] == [
        ("text", "user", "hello"),
        ("text", "assistant", ""),
    ]


def test_build_conversation_with_images_places_them_first():
    img_a, img_b = object(), object()
    parts = build_conversation(prompt="describe", images=[img_a, img_b])
    assert [(p.kind, p.role) for p in parts] == [
        ("image_und", "user"),
        ("image_und", "user"),
        ("text", "user"),
        ("text", "assistant"),
    ]
    assert parts[0].image is img_a
    assert parts[1].image is img_b
    assert parts[2].text == "describe"
    assert parts[3].text == ""


def test_build_conversation_assistant_marker_is_empty_text_part():
    """The trailing assistant marker is the anchor for sampled-token append."""
    parts = build_conversation(prompt="hi")
    assistant_markers = [p for p in parts if p.role == "assistant"]
    assert len(assistant_markers) == 1
    marker = assistant_markers[0]
    assert marker.kind == "text"
    assert marker.text == ""
    assert marker.input_ids is None
    assert marker.inputs_embeds is None


def test_build_conversation_force_image_gen_does_not_change_layout():
    # The flag is consumed upstream via ``request['force_image_gen']``,
    # not by reshaping the conversation — we accept it here purely so
    # callers can forward a single kwargs dict.
    a = build_conversation(prompt="x", force_image_gen=False)
    b = build_conversation(prompt="x", force_image_gen=True)
    assert [(p.kind, p.role) for p in a] == [(p.kind, p.role) for p in b]


# ── ConversationPart shape ───────────────────────────────────────────────────


def test_conversation_part_defaults_are_none_or_empty():
    p = ConversationPart(kind="text", role="user")
    assert p.text is None
    assert p.image is None
    assert p.pixel_values is None
    assert p.token_id is None
    assert p.input_ids is None
    assert p.inputs_embeds is None
    assert p.meta == {}


def test_conversation_part_can_carry_embedded_token():
    embed = torch.zeros(1, 1, 4)
    p = ConversationPart(
        kind="token",
        role="assistant",
        token_id=42,
        input_ids=torch.tensor([[42]]),
        inputs_embeds=embed,
        meta={"source": "tok_decode"},
    )
    assert p.kind == "token"
    assert p.token_id == 42
    assert torch.equal(p.input_ids, torch.tensor([[42]]))
    assert p.inputs_embeds is embed
    assert p.meta == {"source": "tok_decode"}


# ── Traversal helpers ────────────────────────────────────────────────────────


def test_iter_kind_filters_by_kind_in_order():
    parts = [
        ConversationPart(kind="image_und", role="user"),
        ConversationPart(kind="text", role="user", text="a"),
        ConversationPart(kind="image_und", role="user"),
        ConversationPart(kind="text", role="assistant", text=""),
    ]
    images = list(iter_kind(parts, "image_und"))
    texts = list(iter_kind(parts, "text"))
    assert images == [parts[0], parts[2]]
    assert texts == [parts[1], parts[3]]


def test_unembedded_parts_returns_parts_without_inputs_embeds():
    embed = torch.zeros(1, 1, 4)
    parts = [
        ConversationPart(kind="text", role="user", text="a"),
        ConversationPart(kind="text", role="user", text="b", inputs_embeds=embed),
        ConversationPart(kind="text", role="assistant", text=""),
    ]
    remaining = unembedded_parts(parts)
    assert remaining == [parts[0], parts[2]]


def test_latest_assistant_text_token_ids_filters_role_and_kind():
    """Returns only role=assistant kind=token parts produced by the text
    encoder.  VQVAE-emitted parts are explicitly skipped — their
    ``token_id`` is a VQ codebook index in a separate vocab and would
    decode to garbage (or raise) if fed to the text tokenizer."""
    parts = [
        ConversationPart(kind="text", role="user", text="prompt"),
        ConversationPart(kind="token", role="assistant", token_id=5),
        ConversationPart(kind="token", role="assistant", token_id=7),
        # VQVAE-emitted parts must NOT leak into the text decode list.
        ConversationPart(kind="token", role="assistant", token_id=99, meta={"source": "vqvae"}),
        ConversationPart(kind="token", role="system", token_id=1),  # bos — excluded
    ]
    ids = latest_assistant_text_token_ids(parts)
    assert ids == [5, 7]


def test_latest_assistant_text_token_ids_empty_when_no_assistant_tokens():
    parts = [
        ConversationPart(kind="text", role="user", text="hello"),
        ConversationPart(kind="text", role="assistant", text=""),
    ]
    assert latest_assistant_text_token_ids(parts) == []
