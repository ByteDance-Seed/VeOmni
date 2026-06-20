"""Tests for the backend-independent reference oracle contract."""

from __future__ import annotations

import pytest

from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult, normalize_reference_run_result


def test_normalize_accepts_reference_run_result() -> None:
    result = ReferenceRunResult(canonical={"prompt": "x"}, observations={"hidden": [1]})

    assert normalize_reference_run_result(result) is result


def test_normalize_accepts_new_mapping_shape() -> None:
    result = normalize_reference_run_result(
        {
            "canonical": {"prompt": "x"},
            "observations": {"hidden": 1},
            "raw_output": {"raw": True},
        }
    )

    assert result.canonical == {"prompt": "x"}
    assert result.observations == {"hidden": [1]}
    assert result.raw_output == {"raw": True}


def test_normalize_accepts_plain_observation_mapping() -> None:
    result = normalize_reference_run_result({"hidden": 1})

    assert result.canonical == {}
    assert result.observations == {"hidden": [1]}


def test_normalize_rejects_non_mapping_output() -> None:
    with pytest.raises(TypeError, match="Reference runner output"):
        normalize_reference_run_result(object())


def test_normalize_rejects_canonical_mapping_without_observations() -> None:
    with pytest.raises(TypeError, match="must also declare"):
        normalize_reference_run_result({"canonical": {"prompt": "x"}, "hidden": [1]})
