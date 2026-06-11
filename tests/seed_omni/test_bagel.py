"""Durable BAGEL graph-level official parity tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.seed_omni.fixtures.bagel.adapter import (
    adapt_text_only_fixture,
    assert_image_edit_fixture_schema,
    assert_image_gen_fixture_schema,
    assert_text_fixture_schema,
    assert_text_image_fixture_schema,
)
from tests.seed_omni.fixtures.bagel.compare_image_edit import compare_image_edit_graph
from tests.seed_omni.fixtures.bagel.compare_image_gen import compare_image_gen_graph, smoke_image_gen_full_loop_decode
from tests.seed_omni.fixtures.bagel.compare_text_image_und import (
    compare_text_image_und_graph,
    smoke_text_image_raw_graph,
)
from tests.seed_omni.fixtures.bagel.compare_text_only_graph import compare_text_graph


_ENV_PREFIX = "VEOMNI_V2_TEST_BAGEL_"


def _env_name(suffix: str) -> str:
    return f"{_ENV_PREFIX}{suffix}"


def _env_value(suffix: str) -> str | None:
    return os.environ.get(_env_name(suffix))


def _env_flag(suffix: str) -> bool:
    value = _env_value(suffix)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _env_flag("ENABLE_PARITY_CHECK"),
    reason=f"Set {_env_name('ENABLE_PARITY_CHECK')}=1 to run BAGEL official parity checks.",
)


def test_bagel_text_fixture_schema_and_adapter() -> None:
    fixture_path = _env_value("TEXT_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip(f"Set {_env_name('TEXT_PARITY_FIXTURE')} to validate a generated BAGEL text parity fixture.")

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
    fixture_path = _env_value("TEXT_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('TEXT_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL text graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_text_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_interleave_text_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("TEXT_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('TEXT_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL interleave text graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_text_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
        infer_yaml_name="infer_interleave.yaml",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_fixture_schema() -> None:
    fixture_path = _env_value("IMAGE_GEN_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_PARITY_FIXTURE')} to validate a generated BAGEL image-generation fixture."
        )

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_image_gen_fixture_schema(fixture)


def test_bagel_image_generation_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("IMAGE_GEN_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL image-generation graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_gen_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_interleave_image_span_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("IMAGE_GEN_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL interleave image-span graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_gen_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
        infer_yaml_name="infer_interleave.yaml",
        generation_kwargs_override={"infer_mode": "gen"},
    )
    assert report["all_pass"], report


def test_bagel_image_edit_fixture_schema() -> None:
    fixture_path = _env_value("IMAGE_EDIT_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip(f"Set {_env_name('IMAGE_EDIT_PARITY_FIXTURE')} to validate a BAGEL image-edit fixture.")

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_image_edit_fixture_schema(fixture)


def test_bagel_image_edit_vae_context_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("IMAGE_EDIT_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('IMAGE_EDIT_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL input-image VAE-context graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_edit_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_cfg_text_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("IMAGE_GEN_CFG_TEXT_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_CFG_TEXT_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL CFG-text image-generation graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_gen_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_cfg_text_full_loop_decode_smoke() -> None:
    fixture_path = _env_value("IMAGE_GEN_CFG_TEXT_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    max_flow_steps = _env_value("IMAGE_GEN_CFG_TEXT_FULL_LOOP_STEPS")
    if not fixture_path or not model_root or not max_flow_steps:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_CFG_TEXT_PARITY_FIXTURE')}, {_env_name('SPLIT_MODEL_ROOT')}, and "
            f"{_env_name('IMAGE_GEN_CFG_TEXT_FULL_LOOP_STEPS')} to run BAGEL CFG-text full-loop decode smoke."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = smoke_image_gen_full_loop_decode(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
        max_flow_steps=int(max_flow_steps),
    )
    assert report["all_pass"], report


def test_bagel_image_generation_cfg_image_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("IMAGE_GEN_CFG_IMAGE_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_CFG_IMAGE_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL CFG-image graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_gen_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_cfg_image_full_loop_decode_smoke() -> None:
    fixture_path = _env_value("IMAGE_GEN_CFG_IMAGE_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    max_flow_steps = _env_value("IMAGE_GEN_CFG_IMAGE_FULL_LOOP_STEPS")
    if not fixture_path or not model_root or not max_flow_steps:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_CFG_IMAGE_PARITY_FIXTURE')}, {_env_name('SPLIT_MODEL_ROOT')}, and "
            f"{_env_name('IMAGE_GEN_CFG_IMAGE_FULL_LOOP_STEPS')} to run BAGEL CFG-image full-loop decode smoke."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = smoke_image_gen_full_loop_decode(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
        max_flow_steps=int(max_flow_steps),
    )
    assert report["all_pass"], report


def test_bagel_image_generation_cfg_renorm_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("IMAGE_GEN_CFG_RENORM_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_CFG_RENORM_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL CFG renorm graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_image_gen_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_image_generation_full_loop_decode_smoke() -> None:
    fixture_path = _env_value("IMAGE_GEN_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    max_flow_steps = _env_value("IMAGE_GEN_FULL_LOOP_STEPS")
    if not fixture_path or not model_root or not max_flow_steps:
        pytest.skip(
            f"Set {_env_name('IMAGE_GEN_PARITY_FIXTURE')}, {_env_name('SPLIT_MODEL_ROOT')}, and "
            f"{_env_name('IMAGE_GEN_FULL_LOOP_STEPS')} to run BAGEL full-loop decode smoke."
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
    fixture_path = _env_value("TEXT_IMAGE_PARITY_FIXTURE")
    if not fixture_path:
        pytest.skip(f"Set {_env_name('TEXT_IMAGE_PARITY_FIXTURE')} to validate a generated BAGEL text+image fixture.")

    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=False)
    assert_text_image_fixture_schema(fixture)


def test_bagel_text_image_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("TEXT_IMAGE_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('TEXT_IMAGE_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL text+image graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_text_image_und_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report


def test_bagel_text_image_raw_image_graph_matches_official_fixture() -> None:
    fixture_path = _env_value("TEXT_IMAGE_PARITY_FIXTURE")
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not fixture_path or not model_root:
        pytest.skip(
            f"Set {_env_name('TEXT_IMAGE_PARITY_FIXTURE')} and {_env_name('SPLIT_MODEL_ROOT')} "
            "to run BAGEL raw-image text+image graph parity."
        )

    repo_root = Path(__file__).resolve().parents[2]
    report = compare_text_image_und_graph(
        Path(fixture_path),
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
        use_raw_image=True,
    )
    assert report["all_pass"], report


def test_bagel_text_image_raw_e2e_smoke() -> None:
    model_root = _env_value("SPLIT_MODEL_ROOT")
    if not model_root:
        pytest.skip(f"Set {_env_name('SPLIT_MODEL_ROOT')} to run BAGEL raw text/image E2E smoke.")

    repo_root = Path(__file__).resolve().parents[2]
    report = smoke_text_image_raw_graph(
        Path(model_root),
        config_dir=repo_root / "configs/seed_omni/bagel_7b_mot",
    )
    assert report["all_pass"], report
