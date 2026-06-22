from __future__ import annotations

import pytest
import torch

from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.modules.bagel.carrier_updates import (
    append,
    insert_before,
    materialize_carrier_updates,
    meta_patch,
    replace_fields,
    replace_value,
)


def item(
    value: object,
    *,
    type_: str = "text",
    role: str = "user",
    source: str | None = None,
) -> ConversationItem:
    return ConversationItem(type=type_, value=value, role=role, source=source, meta={})


def test_replace_value_updates_item_value() -> None:
    target = item("raw")

    materialize_carrier_updates([[target]], [replace_value(target, torch.tensor([1, 2, 3]))])

    assert torch.equal(target.value, torch.tensor([1, 2, 3]))


def test_replace_fields_updates_basic_item_fields() -> None:
    target = item("raw", type_="image", role="assistant")

    materialize_carrier_updates(
        [[target]],
        [replace_fields(target, type="output", role="dummy", source="bagel_source", meta={"source": "bagel_test"})],
    )

    assert target.type == "output"
    assert target.role == "dummy"
    assert target.source == "bagel_source"
    assert target.meta == {"source": "bagel_test"}


def test_meta_patch_accepts_allowed_fields_and_rejects_unknown_fields() -> None:
    target = item("raw")

    materialize_carrier_updates(
        [[target]],
        [
            meta_patch(
                target,
                {
                    "input_ids": torch.tensor([1]),
                    "labels": torch.tensor([-100]),
                    "attention_mask": torch.tensor([1]),
                    "timestep": torch.tensor([0.5]),
                },
            )
        ],
    )

    assert set(target.meta) == {"input_ids", "labels", "attention_mask", "timestep"}
    with pytest.raises(ValueError, match="Unsupported BAGEL carrier meta"):
        meta_patch(target, {"packed_position_ids": torch.tensor([0])})


def test_insert_before_and_replace_same_original_item_use_snapshot_anchor() -> None:
    before = item("before")
    target = item("target")
    after = item("after")
    inserted = item("inserted", type_="output", role="assistant")
    sample = [before, target, after]

    materialize_carrier_updates(
        [sample],
        [
            insert_before(target, inserted),
            replace_value(target, "updated target"),
        ],
    )

    assert sample == [before, inserted, target, after]
    assert target.value == "updated target"


def test_append_adds_item_to_sample_without_module_anchor() -> None:
    first = item("first")
    sample = [first]
    new_item = item("new", type_="output", role="assistant")

    materialize_carrier_updates([sample], [append(sample, new_item)])

    assert sample == [first, new_item]


def test_incompatible_same_field_updates_fail() -> None:
    target = item("raw")

    with pytest.raises(ValueError, match="same field"):
        materialize_carrier_updates(
            [[target]],
            [
                replace_value(target, "first"),
                replace_value(target, "second"),
            ],
        )


def test_incompatible_same_meta_updates_fail() -> None:
    target = item("raw")

    with pytest.raises(ValueError, match="same meta key"):
        materialize_carrier_updates(
            [[target]],
            [
                meta_patch(target, {"input_ids": torch.tensor([1])}),
                meta_patch(target, {"input_ids": torch.tensor([2])}),
            ],
        )
