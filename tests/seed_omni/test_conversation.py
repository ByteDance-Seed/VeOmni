"""Unit tests for the SeedOmni V2 inference conversation-list helpers."""

from __future__ import annotations

import torch

from veomni.models.seed_omni.conversation import (
    ConversationItem,
    build_conversation,
    collect_modality_batch,
    collect_prompt_embeds,
    get_llm_embed,
    get_token_id,
    is_embedded,
    iter_modality_items,
    iter_type,
    latest_assistant_text_token_ids,
    maybe_merge_outputs,
    seal_outputs,
    unembedded_parts,
)


def test_build_conversation_text_only_yields_two_parts():
    parts = build_conversation(prompt="hello")
    assert [(p.type, p.role, p.value) for p in parts] == [
        ("text", "user", "hello"),
        ("text", "assistant", ""),
    ]


def test_build_conversation_with_images_places_them_first():
    img_a, img_b = object(), object()
    parts = build_conversation(prompt="describe", images=[img_a, img_b])
    assert [(p.type, p.role) for p in parts] == [
        ("image", "user"),
        ("image", "user"),
        ("text", "user"),
        ("text", "assistant"),
    ]
    assert parts[0].value is img_a
    assert parts[1].value is img_b
    assert parts[2].value == "describe"
    assert parts[3].value == ""


def test_build_conversation_assistant_marker_is_empty_text_part():
    parts = build_conversation(prompt="hi")
    assistant_markers = [p for p in parts if p.role == "assistant"]
    assert len(assistant_markers) == 1
    marker = assistant_markers[0]
    assert marker.type == "text"
    assert marker.value == ""
    assert not is_embedded(marker)


def test_conversation_item_defaults():
    p = ConversationItem(type="text", role="user")
    assert p.value is None
    assert p.meta == {}


def test_conversation_item_embedded_token():
    embed = torch.zeros(1, 1, 4)
    p = ConversationItem(
        type="token",
        value=embed,
        role="assistant",
        meta={"token_id": 42, "source": "tok_decode"},
    )
    assert p.type == "token"
    assert get_token_id(p) == 42
    assert torch.equal(get_llm_embed(p), embed)


def test_iter_type_filters_by_type_in_order():
    parts = [
        ConversationItem(type="image", role="user"),
        ConversationItem(type="text", value="a", role="user"),
        ConversationItem(type="image", role="user"),
        ConversationItem(type="text", value="", role="assistant"),
    ]
    images = list(iter_type(parts, "image"))
    texts = list(iter_type(parts, "text"))
    assert images == [parts[0], parts[2]]
    assert texts == [parts[1], parts[3]]


def test_unembedded_parts_returns_parts_without_embed():
    embed = torch.zeros(1, 1, 4)
    parts = [
        ConversationItem(type="text", value="a", role="user"),
        ConversationItem(type="text", value=embed, role="user"),
        ConversationItem(type="text", value="", role="assistant"),
    ]
    remaining = unembedded_parts(parts)
    assert remaining == [parts[0]]


def test_latest_assistant_text_token_ids_filters_role_and_type():
    parts = [
        ConversationItem(type="text", value="prompt", role="user"),
        ConversationItem(type="token", value=5, role="assistant", meta={"token_id": 5}),
        ConversationItem(type="token", value=7, role="assistant", meta={"token_id": 7}),
        ConversationItem(type="token", value=99, role="assistant", meta={"token_id": 99, "source": "vqvae"}),
        ConversationItem(type="token", value=1, role="system", meta={"token_id": 1}),
    ]
    ids = latest_assistant_text_token_ids(parts)
    assert ids == [5, 7]


def test_latest_assistant_text_token_ids_empty_when_no_assistant_tokens():
    parts = [
        ConversationItem(type="text", value="hello", role="user"),
        ConversationItem(type="text", value="", role="assistant"),
    ]
    assert latest_assistant_text_token_ids(parts) == []


def test_is_embedded_recognizes_siglip_patch_tensor():
    """``(1, P, D)`` patch embeds must not be mistaken for ``(1, H, W)`` pixels."""
    embed = torch.zeros(1, 576, 2048)
    item = ConversationItem(type="image", value=embed, role="user")
    assert is_embedded(item)
    assert get_llm_embed(item).shape == (1, 576, 2048)


def test_is_embedded_rejects_chw_pixels():
    pixels = torch.zeros(3, 384, 384)
    item = ConversationItem(type="image", value=pixels, role="user")
    assert not is_embedded(item)


def test_output_merge_and_seal():
    a = ConversationItem(type="output", value=torch.zeros(1, 2, 4), meta={"phase": "text"})
    b = ConversationItem(type="output", value=torch.ones(1, 1, 4), meta={"phase": "text"})
    parts = [a, b]
    assert maybe_merge_outputs(parts, phase="text")
    assert len(parts) == 1
    assert parts[0].value.shape == (1, 3, 4)
    assert seal_outputs(parts, new_type="text") == 1
    assert parts[0].type == "text"
    assert "phase" not in parts[0].meta


def test_collect_prompt_embeds_skips_output():
    embed = torch.zeros(1, 2, 4)
    parts = [
        ConversationItem(type="text", value=embed, role="user"),
        ConversationItem(type="output", value=torch.ones(1, 1, 4), meta={"phase": "text"}),
    ]
    chunks = collect_prompt_embeds(parts)
    assert len(chunks) == 1
    assert torch.equal(chunks[0], embed)


def test_iter_modality_items_allows_multiple_images_per_sample():
    img_a, img_b, img_c = object(), object(), object()
    batch = [
        [
            ConversationItem(type="image", value=img_a, role="user"),
            ConversationItem(type="image", value=img_b, role="user"),
        ],
        [ConversationItem(type="image", value=img_c, role="user")],
    ]
    items = list(iter_modality_items(batch, ["image"], roles=["user"]))
    assert [item.value for item in items] == [img_a, img_b, img_c]
    assert collect_modality_batch(batch, ["image"], roles=["user"]) == [img_a, img_b, img_c]
