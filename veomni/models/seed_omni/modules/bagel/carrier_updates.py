"""BAGEL-local semantic carrier update materializer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from ...conversation import ConversationItem


ALLOWED_META_KEYS = frozenset(
    {
        "input_ids",
        "labels",
        "attention_mask",
        "position_ids",
        "timestep",
        "noise",
        "flow_velocity_target",
        "source",
    }
)
_FIELD_KEYS = frozenset({"type", "value", "role", "source", "meta"})


@dataclass(frozen=True)
class _CarrierUpdate:
    op: str
    item: ConversationItem | None = None
    fields: Mapping[str, Any] | None = None
    patch: Mapping[str, Any] | None = None
    remove: tuple[str, ...] = ()
    new_item: ConversationItem | None = None
    sample: Sequence[ConversationItem] | None = None


def replace_value(item: ConversationItem, value: Any) -> _CarrierUpdate:
    return replace_fields(item, value=value)


def replace_fields(item: ConversationItem, **fields: Any) -> _CarrierUpdate:
    unknown = set(fields) - _FIELD_KEYS
    if unknown:
        raise ValueError(f"Unsupported BAGEL carrier field update(s): {sorted(unknown)}.")
    meta = fields.get("meta")
    if meta is not None:
        _validate_meta_keys(meta.keys())
    return _CarrierUpdate(op="fields", item=item, fields=dict(fields))


def insert_before(item: ConversationItem, new_item: ConversationItem) -> _CarrierUpdate:
    return _CarrierUpdate(op="insert_before", item=item, new_item=new_item)


def insert_after(item: ConversationItem, new_item: ConversationItem) -> _CarrierUpdate:
    return _CarrierUpdate(op="insert_after", item=item, new_item=new_item)


def append(sample: Sequence[ConversationItem], new_item: ConversationItem) -> _CarrierUpdate:
    return _CarrierUpdate(op="append", sample=sample, new_item=new_item)


def meta_patch(
    item: ConversationItem,
    patch: Mapping[str, Any],
    *,
    remove: Iterable[str] = (),
) -> _CarrierUpdate:
    remove_tuple = tuple(remove)
    _validate_meta_keys(patch.keys())
    _validate_meta_keys(remove_tuple)
    overlap = set(patch).intersection(remove_tuple)
    if overlap:
        raise ValueError(f"BAGEL carrier meta patch both sets and removes key(s): {sorted(overlap)}.")
    return _CarrierUpdate(op="meta", item=item, patch=dict(patch), remove=remove_tuple)


def materialize_carrier_updates(
    conversation_list: Sequence[ConversationItem] | Sequence[Sequence[ConversationItem]] | None,
    updates: Iterable[_CarrierUpdate],
) -> None:
    update_list = list(updates)
    if not update_list:
        return

    _validate_update_conflicts(update_list)
    samples = _normalize_samples(conversation_list)
    item_locations, sample_locations = _snapshot_locations(samples)
    structural_updates: list[tuple[int, int, int, ConversationItem]] = []

    for order, update in enumerate(update_list):
        if update.op == "fields":
            assert update.item is not None
            for key, value in (update.fields or {}).items():
                setattr(update.item, key, value)
            continue
        if update.op == "meta":
            assert update.item is not None
            for key in update.remove:
                update.item.meta.pop(key, None)
            for key, value in (update.patch or {}).items():
                update.item.meta[key] = value
            continue

        if update.op == "append":
            if update.sample is None or update.new_item is None:
                raise RuntimeError("BAGEL carrier append update is missing its sample or item.")
            sample_index = sample_locations.get(id(update.sample))
            if sample_index is None:
                raise RuntimeError("BAGEL carrier append sample is not present in the materialization snapshot.")
            if _structural_update_already_materialized(
                samples[sample_index], len(samples[sample_index]), update.new_item
            ):
                continue
            structural_updates.append((sample_index, len(samples[sample_index]), order, update.new_item))
            continue

        if update.item is None or update.new_item is None:
            raise RuntimeError(f"BAGEL carrier {update.op} update is missing its anchor or item.")
        location = item_locations.get(id(update.item))
        if location is None:
            raise RuntimeError("BAGEL carrier update anchor is not present in the materialization snapshot.")
        sample_index, item_index = location
        insert_index = item_index if update.op == "insert_before" else item_index + 1
        if _structural_update_already_materialized(samples[sample_index], insert_index, update.new_item):
            continue
        structural_updates.append((sample_index, insert_index, order, update.new_item))

    for sample_index, insert_index, _, new_item in sorted(
        structural_updates,
        key=lambda entry: (entry[0], entry[1], entry[2]),
        reverse=True,
    ):
        samples[sample_index].insert(insert_index, new_item)


def _structural_update_already_materialized(
    sample: list[ConversationItem],
    insert_index: int,
    new_item: ConversationItem,
) -> bool:
    del insert_index
    return any(existing is new_item for existing in sample)


def _validate_update_conflicts(updates: list[_CarrierUpdate]) -> None:
    seen_fields: dict[tuple[int, str], Any] = {}
    seen_meta: dict[tuple[int, str], Any] = {}
    for update in updates:
        if update.op == "fields":
            assert update.item is not None
            for key, value in (update.fields or {}).items():
                _record_unique(seen_fields, (id(update.item), key), value, f"field {key!r}")
        elif update.op == "meta":
            assert update.item is not None
            for key in update.remove:
                _record_unique(seen_meta, (id(update.item), key), _Removed, f"meta key {key!r}")
            for key, value in (update.patch or {}).items():
                _record_unique(seen_meta, (id(update.item), key), value, f"meta key {key!r}")


def _record_unique(target: dict[tuple[int, str], Any], key: tuple[int, str], value: Any, label: str) -> None:
    existing = target.get(key, _Missing)
    if existing is _Missing:
        target[key] = value
        return
    if not _values_equal(existing, value):
        raise ValueError(f"Incompatible BAGEL carrier updates for the same {label}.")


def _values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is _Removed or right is _Removed:
        return False
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    try:
        return bool(left == right)
    except RuntimeError:
        return False


def _validate_meta_keys(keys: Iterable[str]) -> None:
    unknown = set(keys) - ALLOWED_META_KEYS
    if unknown:
        raise ValueError(f"Unsupported BAGEL carrier meta key(s): {sorted(unknown)}.")


def _normalize_samples(
    conversation_list: Sequence[ConversationItem] | Sequence[Sequence[ConversationItem]] | None,
) -> list[list[ConversationItem]]:
    if conversation_list is None or not conversation_list:
        return []
    first = conversation_list[0]
    if isinstance(first, ConversationItem):
        return [conversation_list]  # type: ignore[list-item]
    return list(conversation_list)  # type: ignore[arg-type]


def _snapshot_locations(
    samples: list[list[ConversationItem]],
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    item_locations: dict[int, tuple[int, int]] = {}
    sample_locations: dict[int, int] = {}
    for sample_index, sample in enumerate(samples):
        sample_locations[id(sample)] = sample_index
        for item_index, item in enumerate(sample):
            item_locations[id(item)] = (sample_index, item_index)
    return item_locations, sample_locations


class _MissingType:
    pass


class _RemovedType:
    pass


_Missing = _MissingType()
_Removed = _RemovedType()


__all__ = [
    "ALLOWED_META_KEYS",
    "append",
    "insert_after",
    "insert_before",
    "materialize_carrier_updates",
    "meta_patch",
    "replace_fields",
    "replace_value",
]
