"""Tests for parity-suite specification parsing."""

from __future__ import annotations

import pytest

from tests.seed_omni.parity_suite.core import DEFAULT_GATE, GateSpec, RecipeSpec, RunSpec


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
    assert run.gate.min_cuda_devices == 2
    assert run.options == {"strategy": "fsdp2", "compare_direct": True, "nproc_per_node": 2}


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
