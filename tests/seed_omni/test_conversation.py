"""Unit tests for the SeedOmni V2 inference conversation-list helpers."""

from __future__ import annotations

import torch

from veomni.models.seed_omni.conversation import (
    ConversationItem,
    build_conversation,
    collect_desired_values,
    iter_desired_items,
    maybe_merge_outputs,
    seal_outputs,
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


def test_conversation_item_defaults():
    p = ConversationItem(type="text", role="user")
    assert p.value is None
    assert p.meta == {}


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


def test_iter_desired_items_allows_multiple_images_per_sample():
    img_a, img_b, img_c = object(), object(), object()
    batch = [
        [
            ConversationItem(type="image", value=img_a, role="user"),
            ConversationItem(type="image", value=img_b, role="user"),
        ],
        [ConversationItem(type="image", value=img_c, role="user")],
    ]
    items = list(iter_desired_items(batch, types=["image"], roles=["user"]))
    assert [item.value for item in items] == [img_a, img_b, img_c]
    assert collect_desired_values(batch, types=["image"], roles=["user"]) == [img_a, img_b, img_c]
