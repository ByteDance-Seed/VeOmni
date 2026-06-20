"""Tests for parity-suite specification parsing."""

from __future__ import annotations

import pytest

from tests.seed_omni.parity_suite.core import (
    DEFAULT_GATE,
    GateSpec,
    LauncherSpec,
    RecipeSpec,
    ReferenceSpec,
    RunSpec,
    V2ModelSpec,
    select_v2_model_target,
)
from tests.seed_omni.parity_suite.core.config.discovery import _enabled_runs, discover_cases
from tests.seed_omni.parity_suite.core.config.spec import TierSelection


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


def test_run_spec_parses_enable_field() -> None:
    run = RunSpec.from_dict(
        recipe_id="toy",
        tier="graph",
        index=0,
        data={"id": "disabled_graph", "enable": False},
    )

    assert run.enable is False
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
    model_gate = GateSpec(discover=False, requires_cuda=True, min_cuda_devices=1)
    recipe_gate = GateSpec(requires_cuda=False, requires_v2_model=False)
    run_gate = GateSpec(min_cuda_devices=2)

    gate = DEFAULT_GATE.merge(model_gate).merge(recipe_gate).merge(run_gate)

    assert gate.discover is False
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


def test_config_bool_fields_reject_string_values(tmp_path) -> None:
    with pytest.raises(TypeError, match="gate.requires_cuda must be a YAML bool"):
        GateSpec.from_dict({"requires_cuda": "false"})
    with pytest.raises(TypeError, match="tiers.module must be a YAML bool"):
        TierSelection.from_dict({"module": "true"})
    with pytest.raises(TypeError, match="launcher.enable_parallel must be a YAML bool"):
        LauncherSpec.from_dict({"enable_parallel": "false"})

    model_dir = tmp_path / "quoted_discover"
    model_dir.mkdir()
    (model_dir / "base.yaml").write_text(
        'discover: "false"\nv2_model:\n  hf_model:\n    config_dir: .\n',
        encoding="utf-8",
    )
    (model_dir / "probes.yaml").write_text("{}", encoding="utf-8")
    (model_dir / "recipes.yaml").write_text(
        """
toy:
  - graph: train
    stimulus: {}
    reference:
      oracle: hf_model
    runs:
      module:
        - id: one_step
""",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="discover must be a YAML bool"):
        discover_cases([model_dir])


def test_recipe_spec_parses_default_graph_and_runs() -> None:
    recipe = RecipeSpec.from_dict(
        "image_gen_base_one_step",
        {
            "stimulus": {"prompt": "A tiny robot."},
            "reference": {"oracle": "hf_model"},
            "runs": {"graph": [{"id": "base_one_step", "probes": ["image.velocity"]}]},
        },
        default_graph="infer_gen",
    )

    assert recipe.id == "image_gen_base_one_step"
    assert recipe.graph == "infer_gen"
    assert recipe.stimulus == {"prompt": "A tiny robot."}
    assert recipe.runs[0].id == "base_one_step"


def test_v2_model_spec_parses_hf_model_and_module_targets(tmp_path) -> None:
    spec = V2ModelSpec.from_dict(
        {
            "model_root": "model_root",
            "hf_model": {"config_dir": "full_config"},
            "hf_module": {
                "text_encoder": {"config_dir": "text_config"},
                "siglip_navit": {"config_dir": str(tmp_path / "vision_config")},
            },
        },
        repo_root=tmp_path,
    )

    assert spec.hf_model is not None
    assert spec.model_root == tmp_path / "model_root"
    assert spec.hf_model.model_root == tmp_path / "model_root"
    assert spec.hf_model.config_dir == tmp_path / "full_config"
    assert spec.hf_module["text_encoder"].model_root == tmp_path / "model_root"
    assert spec.hf_module["text_encoder"].config_dir == tmp_path / "text_config"
    assert spec.hf_module["siglip_navit"].config_dir == tmp_path / "vision_config"


def test_reference_spec_parses_hf_module_name_list(tmp_path) -> None:
    spec = ReferenceSpec.from_dict(
        {
            "hf_model": {"module": "tests.fake:Reference", "checkpoint": "ckpt"},
            "hf_module": ["text_encoder", "siglip_navit"],
        },
        repo_root=tmp_path,
    )

    assert spec.hf_model is not None
    assert spec.hf_model.checkpoint == tmp_path / "ckpt"
    assert spec.hf_module == ("text_encoder", "siglip_navit")


def test_reference_spec_rejects_hf_module_without_hf_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires reference.hf_model"):
        ReferenceSpec.from_dict({"hf_module": ["text_encoder"]}, repo_root=tmp_path)


def test_v2_model_target_selection_follows_reference_oracle(tmp_path) -> None:
    spec = V2ModelSpec.from_dict(
        {
            "hf_model": {"config_dir": "full_config"},
            "hf_module": {"text_encoder": {"config_dir": "text_config"}},
        },
        repo_root=tmp_path,
    )

    assert (
        select_v2_model_target(spec, "hf_module.text_encoder", model_name="toy", recipe_id="text").config_dir
        == tmp_path / "text_config"
    )
    assert select_v2_model_target(spec, "hf_model", model_name="toy", recipe_id="full").config_dir == (
        tmp_path / "full_config"
    )


def test_recipe_spec_accepts_v2_model_module_overrides(tmp_path) -> None:
    recipe = RecipeSpec.from_dict(
        "text_encoder",
        {
            "stimulus": {"prompt": "hello"},
            "reference": {"oracle": "hf_module.text_encoder"},
            "v2_model": {"module_overrides": {"bagel_vae": {"device": "cpu", "dtype": "float32"}}},
            "runs": {"module": [{"id": "encode_decode"}]},
        },
        default_graph="train",
        repo_root=tmp_path,
    )

    assert recipe.v2_model.module_overrides == {"bagel_vae": {"device": "cpu", "dtype": "float32"}}


def test_recipe_spec_accepts_conversation_stimulus_shapes() -> None:
    single = RecipeSpec.from_dict(
        "text_encoder",
        {
            "stimulus": {
                "conversation_list": [
                    {
                        "type": "text",
                        "value": {"kind": "tensor", "tensor": [1, 2], "dtype": "long"},
                    },
                    {
                        "type": "image",
                        "value": {"kind": "image", "width": 64, "height": 48},
                    },
                ]
            },
            "reference": {"oracle": "hf_module.text_encoder"},
            "runs": {"module": [{"id": "encode"}]},
        },
        default_graph="train",
    )
    batched = RecipeSpec.from_dict(
        "text_encoder",
        {
            "stimulus": {
                "batched_conversation_list": [
                    [
                        {
                            "type": "text",
                            "value": {"kind": "tensor", "tensor": [1], "dtype": "long"},
                        }
                    ],
                    [
                        {
                            "type": "image",
                            "value": {"kind": "random", "shape": [3, 4, 4], "seed": 7},
                        }
                    ],
                ]
            },
            "reference": {"oracle": "hf_module.text_encoder"},
            "runs": {"module": [{"id": "encode"}]},
        },
        default_graph="train",
    )

    assert single.stimulus["conversation_list"][0]["value"]["kind"] == "tensor"
    assert single.stimulus["conversation_list"][1]["value"]["kind"] == "image"
    assert len(batched.stimulus["batched_conversation_list"]) == 2


def test_recipe_spec_rejects_invalid_conversation_stimulus_shapes() -> None:
    cases = [
        (
            {"conversation_list": [], "batched_conversation_list": []},
            ValueError,
            "only one of conversation_list or batched_conversation_list",
        ),
        (
            {
                "conversation_list": [
                    [
                        {
                            "type": "text",
                            "value": {"kind": "tensor", "tensor": [1], "dtype": "long"},
                        }
                    ]
                ]
            },
            TypeError,
            "Use stimulus.batched_conversation_list",
        ),
        (
            {
                "conversation_list": [
                    {
                        "type": "latent",
                        "value": {"kind": "tensor", "tensor": [1.0], "dtype": "float"},
                    }
                ]
            },
            ValueError,
            r"type must be one of .*'image'.*'output'.*'text'",
        ),
    ]

    for stimulus, error_type, message in cases:
        with pytest.raises(error_type, match=message):
            RecipeSpec.from_dict(
                "bad",
                {
                    "stimulus": stimulus,
                    "reference": {"oracle": "hf_module.text_encoder"},
                    "runs": {"module": [{"id": "encode"}]},
                },
                default_graph="train",
            )


def test_discover_cases_skips_model_with_discover_false(tmp_path) -> None:
    model_dir = tmp_path / "archived"
    model_dir.mkdir()
    (model_dir / "base.yaml").write_text(
        """
name: archived
v2_model:
  hf_model:
    config_dir: .
gate:
  discover: false
""",
        encoding="utf-8",
    )
    (model_dir / "probes.yaml").write_text("{}", encoding="utf-8")
    (model_dir / "recipes.yaml").write_text(
        """
toy:
  - graph: train
    stimulus: {}
    reference:
      oracle: hf_model
      kind: toy
    runs:
      module:
        - id: skipped
""",
        encoding="utf-8",
    )

    assert discover_cases([model_dir]) == ()


def test_enabled_runs_skips_disabled_runs() -> None:
    recipe = RecipeSpec.from_dict(
        "image_gen_base_one_step",
        {
            "stimulus": {"prompt": "A tiny robot."},
            "reference": {"oracle": "hf_model"},
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


def test_recipe_spec_validates_graph_field() -> None:
    with pytest.raises(ValueError, match="must declare graph"):
        RecipeSpec.from_dict(
            "image_gen_base_one_step",
            {
                "runs": {"graph": [{"id": "base_one_step"}]},
            },
        )
    with pytest.raises(ValueError, match="expected 'infer_gen'"):
        RecipeSpec.from_dict(
            "image_gen_base_one_step",
            {
                "graph": "infer_und",
                "runs": {"graph": [{"id": "base_one_step"}]},
            },
            default_graph="infer_gen",
        )
