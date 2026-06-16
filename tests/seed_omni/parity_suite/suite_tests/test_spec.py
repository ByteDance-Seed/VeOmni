"""Tests for parity-suite specification parsing."""

from __future__ import annotations

import pytest

from tests.seed_omni.parity_suite.core import DEFAULT_GATE, GateSpec, LauncherSpec, RecipeSpec, RunSpec
from tests.seed_omni.parity_suite.core.config.discovery import _enabled_runs


def test_run_spec_keeps_kind_specific_settings_under_options() -> None:
    run = RunSpec.from_dict(
        recipe_id="toy",
        tier="framework",
        index=0,
        data={
            "id": "fsdp2_numeric",
            "kind": "distributed_train",
            "gate": {"min_cuda_devices": 2},
            "options": {
                "strategy": "fsdp2",
                "compare_direct": True,
                "nproc_per_node": 2,
            },
        },
    )

    assert run.id == "fsdp2_numeric"
    assert run.kind == "distributed_train"
    assert run.enable is True
    assert run.gate.min_cuda_devices == 2
    assert run.options == {"strategy": "fsdp2", "compare_direct": True, "nproc_per_node": 2}


def test_run_spec_accepts_explicit_disable() -> None:
    run = RunSpec.from_dict(
        recipe_id="toy",
        tier="graph",
        index=0,
        data={"id": "disabled_graph", "enable": False},
    )

    assert run.enable is False


def test_run_spec_rejects_non_bool_enable() -> None:
    with pytest.raises(TypeError, match="enable must be a bool"):
        RunSpec.from_dict(
            recipe_id="toy",
            tier="graph",
            index=0,
            data={"id": "disabled_graph", "enable": "false"},
        )


def test_run_spec_rejects_implicit_kind_specific_fields() -> None:
    with pytest.raises(ValueError, match="Put tier- or kind-specific settings under options"):
        RunSpec.from_dict(
            recipe_id="toy",
            tier="framework",
            index=0,
            data={
                "id": "fsdp2_numeric",
                "kind": "distributed_train",
                "strategy": "fsdp2",
            },
        )


def test_gate_spec_merges_bool_overrides_and_device_floor() -> None:
    model_gate = GateSpec(requires_cuda=True, min_cuda_devices=1)
    recipe_gate = GateSpec(requires_cuda=False, requires_v2_model=False)
    run_gate = GateSpec(min_cuda_devices=2)

    gate = DEFAULT_GATE.merge(model_gate).merge(recipe_gate).merge(run_gate)

    assert gate.requires_parity_env is True
    assert gate.requires_cuda is False
    assert gate.requires_reference_checkpoint is True
    assert gate.requires_v2_model is False
    assert gate.min_cuda_devices == 2


def test_launcher_spec_defaults_to_serial_and_optional_device_cap() -> None:
    assert LauncherSpec.from_dict(None) == LauncherSpec()

    launcher = LauncherSpec.from_dict({"enable_parallel": True, "max_cuda_devices": 8})

    assert launcher.enable_parallel is True
    assert launcher.max_cuda_devices == 8


def test_recipe_spec_parses_default_graph_and_runs() -> None:
    recipe = RecipeSpec.from_dict(
        "image_gen_base_one_step",
        {
            "stimulus": {"prompt": "A tiny robot."},
            "runs": {"graph": [{"id": "base_one_step", "probes": ["image.velocity"]}]},
        },
        default_graph="infer_gen",
    )

    assert recipe.id == "image_gen_base_one_step"
    assert recipe.graph == "infer_gen"
    assert recipe.stimulus == {"prompt": "A tiny robot."}
    assert recipe.runs[0].id == "base_one_step"


def test_enabled_runs_skips_disabled_runs() -> None:
    recipe = RecipeSpec.from_dict(
        "image_gen_base_one_step",
        {
            "stimulus": {"prompt": "A tiny robot."},
            "runs": {
                "graph": [
                    {"id": "enabled_graph"},
                    {"id": "disabled_graph", "enable": False},
                ],
                "module": [{"id": "disabled_by_tier"}],
            },
        },
        default_graph="infer_gen",
    )

    runs = _enabled_runs(("graph",), recipe)

    assert [run.id for run in runs] == ["enabled_graph"]


def test_recipe_spec_requires_graph_without_default() -> None:
    with pytest.raises(ValueError, match="must declare graph"):
        RecipeSpec.from_dict(
            "image_gen_base_one_step",
            {
                "runs": {"graph": [{"id": "base_one_step"}]},
            },
        )


def test_recipe_spec_rejects_graph_that_conflicts_with_default() -> None:
    with pytest.raises(ValueError, match="expected 'infer_gen'"):
        RecipeSpec.from_dict(
            "image_gen_base_one_step",
            {
                "graph": "infer_und",
                "runs": {"graph": [{"id": "base_one_step"}]},
            },
            default_graph="infer_gen",
        )
