"""``OmniArguments`` goes through the shared argparse layer in ``veomni.arguments.parser``.

That layer unwraps ``typing.Optional[X]`` only. A PEP 604 ``X | None`` is a
``types.UnionType``: it is handed to argparse as the ``type`` callable, and the
flag fails with ``'types.UnionType' object is not callable``. So every field in
the tree is written ``Optional[X]``, as in the single-model arguments.
"""

import dataclasses
import sys
import types
import typing

import pytest

from veomni.arguments.omni_arguments_types import OmniArguments, OmniModuleRuntimeArguments
from veomni.arguments.omni_parser import parse_omni_args


def _contains_pep604_union(annotation) -> bool:
    """Also inside ``list[...]``: the parser hands a list's item type to argparse too."""
    if isinstance(annotation, types.UnionType):
        return True
    return any(_contains_pep604_union(arg) for arg in typing.get_args(annotation))


def _pep604_fields(cls, seen=None):
    seen = set() if seen is None else seen
    if cls in seen:
        return []
    seen.add(cls)
    hints = typing.get_type_hints(cls)
    found = []
    for field in dataclasses.fields(cls):
        field_type = hints.get(field.name, field.type)
        if _contains_pep604_union(field_type):
            found.append(f"{cls.__name__}.{field.name}: {field_type}")
        if typing.get_origin(field_type) in (typing.Union, types.UnionType):
            field_type = next(arg for arg in typing.get_args(field_type) if arg is not type(None))
        if dataclasses.is_dataclass(field_type):
            found += _pep604_fields(field_type, seen)
    return found


@pytest.mark.parametrize("root", [OmniArguments, OmniModuleRuntimeArguments])
def test_no_argument_field_is_written_as_a_pep604_union(root):
    """Per-module overlays are instantiated by the same layer, from a dict the
    ``OmniArguments`` walk does not descend into."""
    assert _pep604_fields(root) == []


def test_an_optional_field_parses_from_the_cli(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--model.model_path", "/tmp/fake_omni", "--data.train_path", "t.jsonl", "--train.max_steps", "5"],
    )

    assert parse_omni_args(OmniArguments).train.max_steps == 5
