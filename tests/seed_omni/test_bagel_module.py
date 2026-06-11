"""Temporary BAGEL module-level parity tests.

Remove this file once the covered module checks are replaced by graph-level
tests in ``test_bagel.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.seed_omni.fixtures.bagel.compare_image_gen_module import (
    assert_image_gen_fixture_schema,
    compare_image_gen_module,
)


def test_bagel_image_generation_fixture_schema() -> None:
    fixture_path = os.environ.get("BAGEL_IMAGE_GEN_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip("Set BAGEL_IMAGE_GEN_PARITY_FIXTURE to validate a generated BAGEL image-generation fixture.")

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_image_gen_fixture_schema(fixture)


def test_bagel_image_generation_modules_match_official_fixture() -> None:
    fixture_path = os.environ.get("BAGEL_IMAGE_GEN_PARITY_FIXTURE")
    model_root = os.environ.get("BAGEL_SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            "Set BAGEL_IMAGE_GEN_PARITY_FIXTURE and BAGEL_SPLIT_MODEL_ROOT to run BAGEL image-generation module parity."
        )

    report = compare_image_gen_module(Path(fixture_path), Path(model_root))
    assert report["all_pass"], report
