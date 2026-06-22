"""Shared stimulus parsing and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# Stimulus schema constants ----------------------------------------------------


CONVERSATION_ITEM_TYPES = frozenset({"image", "text", "output"})
CONVERSATION_VALUE_KINDS = frozenset({"tensor", "random", "linspace", "image"})


# Public stimulus helpers ------------------------------------------------------


def conversation_stimulus_to_batched_specs(stimulus: Mapping[str, Any]) -> Any | None:
    """Return authored conversation stimulus as an explicit batch, if present."""

    if "conversation_list" in stimulus and "batched_conversation_list" in stimulus:
        raise ValueError("stimulus.conversation_list and stimulus.batched_conversation_list are mutually exclusive.")

    if "conversation_list" in stimulus:
        conversation_list = stimulus["conversation_list"]
        _validate_sequence(conversation_list, "stimulus.conversation_list")
        return [conversation_list]

    if "batched_conversation_list" in stimulus:
        batched_conversation_list = stimulus["batched_conversation_list"]
        _validate_sequence(batched_conversation_list, "stimulus.batched_conversation_list")
        return batched_conversation_list

    return None


def validate_stimulus(recipe_label: str, stimulus: Mapping[str, Any]) -> None:
    """Validate suite-level recipe stimulus fields that are shared across runtimes."""

    has_conversation_list = "conversation_list" in stimulus
    has_batched_conversation_list = "batched_conversation_list" in stimulus
    if has_conversation_list and has_batched_conversation_list:
        raise ValueError(
            f"{recipe_label} stimulus must declare only one of conversation_list or batched_conversation_list."
        )
    if has_conversation_list:
        _validate_conversation_items(
            recipe_label,
            stimulus["conversation_list"],
            path="stimulus.conversation_list",
        )
    if has_batched_conversation_list:
        batched = stimulus["batched_conversation_list"]
        if not isinstance(batched, list):
            raise TypeError(f"{recipe_label} stimulus.batched_conversation_list must be a list of samples.")
        for sample_index, sample in enumerate(batched):
            _validate_conversation_items(
                recipe_label,
                sample,
                path=f"stimulus.batched_conversation_list[{sample_index}]",
            )


# Internal validation helpers --------------------------------------------------


def _validate_sequence(value: Any, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a YAML sequence.")


def _validate_conversation_items(recipe_label: str, items: Any, *, path: str) -> None:
    if not isinstance(items, list):
        raise TypeError(f"{recipe_label} {path} must be a list of item specs.")
    for item_index, item in enumerate(items):
        item_path = f"{path}[{item_index}]"
        if not isinstance(item, dict):
            raise TypeError(
                f"{recipe_label} {item_path} must be an item mapping. "
                "Use stimulus.batched_conversation_list for explicit batches."
            )
        item_type = str(item.get("type", "text"))
        if item_type not in CONVERSATION_ITEM_TYPES:
            raise ValueError(
                f"{recipe_label} {item_path}.type must be one of {sorted(CONVERSATION_ITEM_TYPES)}; got {item_type!r}."
            )
        if "value" not in item:
            raise ValueError(f"{recipe_label} {item_path} must declare value.")
        _validate_conversation_value(recipe_label, item["value"], item_type=item_type, path=f"{item_path}.value")


def _validate_conversation_value(recipe_label: str, value: Any, *, item_type: str, path: str) -> None:
    if item_type == "text":
        if not isinstance(value, str):
            raise TypeError(f"{recipe_label} {path} for text items must be a string.")
        return
    if not isinstance(value, dict):
        raise TypeError(f"{recipe_label} {path} must be a mapping with a kind field.")
    kind = value.get("kind")
    if kind is None:
        untagged = sorted(key for key in ("tensor", "random", "path") if key in value)
        hint = f" Untagged key(s) are not supported: {untagged}." if untagged else ""
        raise ValueError(f"{recipe_label} {path} must declare kind.{hint}")
    kind = str(kind)
    if kind not in CONVERSATION_VALUE_KINDS:
        raise ValueError(
            f"{recipe_label} {path}.kind must be one of {sorted(CONVERSATION_VALUE_KINDS)}; got {kind!r}."
        )
    if kind == "tensor" and "tensor" not in value:
        raise ValueError(f"{recipe_label} {path} with kind: tensor must declare tensor.")
    if kind == "random" and "shape" not in value:
        raise ValueError(f"{recipe_label} {path} with kind: random must declare shape.")
    if kind == "linspace" and ("start" not in value or "end" not in value):
        raise ValueError(f"{recipe_label} {path} with kind: linspace must declare start and end.")
    if kind == "image":
        for dim in ("width", "height"):
            if dim in value and not isinstance(value[dim], int):
                raise TypeError(f"{recipe_label} {path}.{dim} with kind: image must be an int.")


__all__ = [
    "CONVERSATION_ITEM_TYPES",
    "CONVERSATION_VALUE_KINDS",
    "conversation_stimulus_to_batched_specs",
    "validate_stimulus",
]
