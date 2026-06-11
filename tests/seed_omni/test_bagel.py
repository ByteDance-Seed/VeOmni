"""Durable BAGEL graph-level official parity tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.seed_omni.fixtures.bagel.adapter import (
    adapt_text_only_fixture,
    assert_image_gen_fixture_schema,
    assert_text_fixture_schema,
    assert_text_image_fixture_schema,
)
from tests.seed_omni.fixtures.bagel.compare_image_gen import compare_image_gen_graph, smoke_image_gen_full_loop_decode
from tests.seed_omni.fixtures.bagel.compare_text_image_und import compare_text_image_und_graph
from tests.seed_omni.fixtures.bagel.compare_text_only_graph import compare_text_graph


def test_bagel_text_fixture_schema_and_adapter() -> None:
    fixture_path = os.environ.get("BAGEL_TEXT_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip("Set BAGEL_TEXT_PARITY_FIXTURE to validate a generated BAGEL text parity fixture.")

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_text_fixture_schema(fixture)

    conversation = adapt_text_only_fixture(fixture)
    assert len(conversation) == 1
    item = conversation[0]
    assert item.type == "text"
    assert item.role == "user"
    assert item.source == "bagel_official_fixture"
    assert item.meta["bagel_role"] == "text"
    assert item.meta["raw_text"] == fixture["raw_input"]["prompt"]
    assert torch.equal(item.value, fixture["prepared"]["prompt"]["packed_text_ids"])
    assert torch.equal(item.meta["expected"]["greedy_token"], fixture["one_step"]["greedy_token"])


def test_bagel_text_graph_matches_official_fixture() -> None:
    fixture_path = os.environ.get("BAGEL_TEXT_PARITY_FIXTURE")
    model_root = os.environ.get("BAGEL_SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip("Set BAGEL_TEXT_PARITY_FIXTURE and BAGEL_SPLIT_MODEL_ROOT to run BAGEL text graph parity.")

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_text_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_fixture_schema() -> None:
    fixture_path = os.environ.get("BAGEL_IMAGE_GEN_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip("Set BAGEL_IMAGE_GEN_PARITY_FIXTURE to validate a generated BAGEL image-generation fixture.")

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_image_gen_fixture_schema(fixture)


def test_bagel_image_generation_graph_matches_official_fixture() -> None:
    fixture_path = os.environ.get("BAGEL_IMAGE_GEN_PARITY_FIXTURE")
    model_root = os.environ.get("BAGEL_SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            "Set BAGEL_IMAGE_GEN_PARITY_FIXTURE and BAGEL_SPLIT_MODEL_ROOT to run BAGEL image-generation graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_gen_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_full_loop_decode_smoke() -> None:
    fixture_path = os.environ.get("BAGEL_IMAGE_GEN_PARITY_FIXTURE")
    model_root = os.environ.get("BAGEL_SPLIT_MODEL_ROOT")
    max_flow_steps = os.environ.get("BAGEL_IMAGE_GEN_FULL_LOOP_STEPS")
    if not fixture_path or not model_root or not max_flow_steps:
        pytest.skip(
            "Set BAGEL_IMAGE_GEN_PARITY_FIXTURE, BAGEL_SPLIT_MODEL_ROOT, and "
            "BAGEL_IMAGE_GEN_FULL_LOOP_STEPS to run BAGEL full-loop decode smoke."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = smoke_image_gen_full_loop_decode(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
        max_flow_steps=int(max_flow_steps),
    )
    assert report["all_pass"], report


def test_bagel_text_image_fixture_schema() -> None:
    fixture_path = os.environ.get("BAGEL_TEXT_IMAGE_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip("Set BAGEL_TEXT_IMAGE_PARITY_FIXTURE to validate a generated BAGEL text+image fixture.")

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_text_image_fixture_schema(fixture)


def test_bagel_text_image_graph_matches_official_fixture() -> None:
    fixture_path = os.environ.get("BAGEL_TEXT_IMAGE_PARITY_FIXTURE")
    model_root = os.environ.get("BAGEL_SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            "Set BAGEL_TEXT_IMAGE_PARITY_FIXTURE and BAGEL_SPLIT_MODEL_ROOT to run BAGEL text+image graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_text_image_und_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report
