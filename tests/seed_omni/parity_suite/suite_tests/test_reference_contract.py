"""Tests for the reference handler output contract."""

from __future__ import annotations

import pytest

from tests.seed_omni.parity_suite.reference.contract import (
    canonical_from_reference_output,
    is_reference_run_output,
    make_reference_run_output,
)


def test_make_reference_run_output_shapes_contract_payload() -> None:
    output = make_reference_run_output({"prompt": "x"}, {"hidden_state": 1})
    assert output == {"canonical": {"prompt": "x"}, "reference": {"hidden_state": 1}}
    assert is_reference_run_output(output)


def test_canonical_from_reference_output_returns_canonical() -> None:
    canonical = {"prompt": "hello", "kv_lens_after_prompt": [3]}
    reference = {"hidden_state": "tensor"}
    output = make_reference_run_output(canonical, reference)
    assert canonical_from_reference_output(output) == canonical


def test_canonical_from_reference_output_none_returns_empty_mapping() -> None:
    assert canonical_from_reference_output(None) == {}


def test_canonical_from_reference_output_missing_canonical_raises() -> None:
    with pytest.raises(TypeError, match='missing required key "canonical"'):
        canonical_from_reference_output({"reference": {}})


def test_canonical_from_reference_output_missing_reference_raises() -> None:
    with pytest.raises(TypeError, match='missing required key "reference"'):
        canonical_from_reference_output({"canonical": {}})


def test_canonical_from_reference_output_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match='must be a mapping shaped as {"canonical": ..., "reference": ...}'):
        canonical_from_reference_output("not-a-mapping")


def test_canonical_from_reference_output_rejects_non_mapping_values() -> None:
    with pytest.raises(TypeError, match='key "canonical" must be a mapping'):
        canonical_from_reference_output({"canonical": "bad", "reference": {}})
    with pytest.raises(TypeError, match='key "reference" must be a mapping'):
        canonical_from_reference_output({"canonical": {}, "reference": []})


def test_is_reference_run_output_rejects_partial_or_invalid_values() -> None:
    assert not is_reference_run_output(None)
    assert not is_reference_run_output({"canonical": {}})
    assert not is_reference_run_output({"reference": {}})
    assert not is_reference_run_output(42)
