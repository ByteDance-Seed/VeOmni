"""Tests for shared parity-suite runtime helpers."""

from __future__ import annotations

import pytest
import torch

from tests.seed_omni.parity_suite.core.runtime import resolve_torch_dtype


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, torch.float32),
        (torch.float16, torch.float16),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_resolve_torch_dtype_accepts_known_aliases(raw: object, expected: torch.dtype) -> None:
    assert resolve_torch_dtype(raw) is expected


def test_resolve_torch_dtype_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="Unsupported reference dtype"):
        resolve_torch_dtype("fp8")
