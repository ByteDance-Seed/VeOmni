"""Inspect a BAGEL text-only official fixture through the V2 adapter boundary."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import adapt_text_only_fixture, assert_text_fixture_schema  # noqa: E402


def _shape(value: Any) -> Any:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    assert_text_fixture_schema(fixture)
    conversation = adapt_text_only_fixture(fixture)
    item = conversation[0]

    report = {
        "case_id": fixture["metadata"]["case_id"],
        "dtype": fixture["metadata"]["dtype"],
        "prompt": fixture["raw_input"]["prompt"],
        "conversation_items": len(conversation),
        "item": {
            "type": item.type,
            "role": item.role,
            "source": item.source,
            "value_shape": _shape(item.value),
            "expected": _shape(item.meta["expected"]),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
