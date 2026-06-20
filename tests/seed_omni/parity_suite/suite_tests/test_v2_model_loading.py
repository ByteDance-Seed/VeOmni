"""Tests for shared graph-driven V2 model loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.seed_omni.parity_suite.core import NodeSpec
from tests.seed_omni.parity_suite.core.config.discovery import discover_cases
from tests.seed_omni.parity_suite.v2.model import (
    apply_v2_module_override,
    graph_active_module_names,
    load_graph_active_omni_config,
    load_graph_active_omni_modules,
    v2_module_override_map,
    v2_module_target,
)


def test_inference_config_filters_to_graph_active_modules_without_training_graph(tmp_path: Path) -> None:
    config_dir = _write_toy_config(tmp_path)
    case = _case(
        config_dir,
        graph_name="infer_toy",
        graph_domain="inference",
        nodes=(
            NodeSpec(name="encode", module="vision", method="generate", graph="infer_toy", state="prompt"),
            NodeSpec(name="decode", module="text", method="generate", graph="infer_toy", state="prompt"),
        ),
    )

    config = load_graph_active_omni_config(case)

    assert config.module_names == ["text", "vision"]
    assert config.training_graph == []
    assert config.generation_graph == {
        "initial": "prompt",
        "states": {
            "prompt": {
                "body": [{"from": "vision", "to": "text"}, {"from": "text", "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }
    assert config.generation_kwargs == {"max_new_tokens": 1}


def test_training_config_filters_modules_and_keeps_training_graph(tmp_path: Path) -> None:
    config_dir = _write_toy_config(tmp_path)
    case = _case(
        config_dir,
        graph_name="train",
        graph_domain="training",
        nodes=(
            NodeSpec(name="vision", module="vision", method="forward", graph="train", state="train"),
            NodeSpec(name="text", module="text", method="forward", graph="train", state="train"),
        ),
    )

    config = load_graph_active_omni_config(case)

    assert config.module_names == ["text", "vision"]
    assert config.training_graph == [{"from": "vision", "to": "text"}, {"from": "text", "to": "end"}]
    assert config.generation_graph is None


def test_bagel_module_recipe_selects_module_level_config_from_oracle_target() -> None:
    [case] = [
        case
        for case in discover_cases([Path("tests/seed_omni/bagel")])
        if case.recipe.id == "module_text_encoder_prompt" and case.tier == "module" and case.run.id == "prompt_encode"
    ]

    module_names = graph_active_module_names(case)
    config = load_graph_active_omni_config(case)

    assert case.v2_model.config_dir.match("*/tests/seed_omni/bagel/configs/text_encoder")
    assert case.v2_model.model_root.match("*/models/seed_omni/BAGEL-7B-MoT")
    assert module_names == frozenset({"bagel_text_encoder"})
    assert config.module_names == ["bagel_text_encoder"]
    assert config.modules["bagel_text_encoder"]["model"]["model_path"] == "bagel_text_encoder"
    assert config.training_graph == []
    assert config.generation_graph["initial"] == "prompt_encode"
    assert config.generation_graph["states"]["prompt_encode"]["body"] == [
        {"from": "bagel_text_encoder.prompt_encode", "to": "end"}
    ]


def test_bagel_infer_edit_case_declares_recipe_level_v2_module_override() -> None:
    [case] = [
        case
        for case in discover_cases([Path("tests/seed_omni/bagel")])
        if case.recipe.id == "image_text_edit_cfg_disabled"
        and case.tier == "graph"
        and case.run.id == "denoise_one_step"
    ]

    assert case.recipe.v2_model.module_overrides == {"bagel_vae": {"device": "cpu", "dtype": "float32"}}
    assert case.run.options == {"max_tensor_numel": 4000000}


def test_v2_module_loader_reads_recipe_level_module_overrides_by_default(tmp_path: Path) -> None:
    config_dir = _write_toy_config(tmp_path)
    case = _case(
        config_dir,
        graph_name="infer_toy",
        graph_domain="inference",
        nodes=(
            NodeSpec(name="encode", module="vision", method="generate", graph="infer_toy", state="prompt"),
            NodeSpec(name="decode", module="text", method="generate", graph="infer_toy", state="prompt"),
        ),
    )
    case.recipe = SimpleNamespace(
        v2_model=SimpleNamespace(module_overrides={"inactive": {"device": "cpu", "dtype": "float32"}})
    )

    with pytest.raises(KeyError, match=r"v2_model\.module_overrides references inactive module"):
        load_graph_active_omni_modules(
            case,
            ("text", "vision"),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_v2_module_loader_allows_module_tier_override_for_current_module(tmp_path: Path, monkeypatch) -> None:
    config_dir = _write_toy_config(tmp_path)
    case = _case(
        config_dir,
        graph_name="infer_toy",
        graph_domain="inference",
        nodes=(NodeSpec(name="decode", module="text", method="generate", graph="infer_toy", state="prompt"),),
    )
    case.run = SimpleNamespace(tier="module")
    case.recipe = SimpleNamespace(
        v2_model=SimpleNamespace(module_overrides={"text": {"device": "cpu", "dtype": "bfloat16"}})
    )
    loaded_targets = {}

    def fake_load_module(module_name, module_config, *, seed, device, dtype):
        loaded_targets[module_name] = (device, dtype)
        return torch.nn.Identity()

    monkeypatch.setattr(
        "tests.seed_omni.parity_suite.v2.model.load_omni_module_from_parity_config",
        fake_load_module,
    )

    modules = load_graph_active_omni_modules(
        case,
        ("text",),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert set(modules) == {"text"}
    assert loaded_targets == {"text": (torch.device("cpu"), torch.bfloat16)}


def test_v2_module_loader_rejects_recipe_overrides_outside_supported_tiers(tmp_path: Path) -> None:
    config_dir = _write_toy_config(tmp_path)
    case = _case(
        config_dir,
        graph_name="infer_toy",
        graph_domain="inference",
        nodes=(NodeSpec(name="decode", module="text", method="generate", graph="infer_toy", state="prompt"),),
    )
    case.run = SimpleNamespace(tier="reference")
    case.recipe = SimpleNamespace(
        v2_model=SimpleNamespace(module_overrides={"text": {"device": "cpu", "dtype": "float32"}})
    )

    with pytest.raises(ValueError, match="only supported for graph/framework tiers"):
        load_graph_active_omni_modules(
            case,
            ("text",),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_v2_module_override_target_can_change_device_and_dtype() -> None:
    device, dtype = v2_module_target(
        {"device": "cpu", "dtype": "float32"},
        default_device=torch.device("cuda"),
        default_dtype=torch.bfloat16,
    )

    assert device == torch.device("cpu")
    assert dtype == torch.float32


def test_v2_module_override_can_patch_module_config() -> None:
    module = torch.nn.Identity()
    module.config = SimpleNamespace(min_image_size=512, max_image_size=1024)

    apply_v2_module_override(module, {"config": {"min_image_size": 32, "max_image_size": 32}})

    assert module.config.min_image_size == 32
    assert module.config.max_image_size == 32


def test_v2_module_override_map_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="v2_model.module_overrides must be a mapping"):
        v2_module_override_map(["bagel_vae"])


def _case(
    config_dir: Path,
    *,
    graph_name: str,
    graph_domain: str,
    nodes: tuple[NodeSpec, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            name="toy",
            seed=7,
            v2_model=SimpleNamespace(
                config_dir=config_dir,
                model_root=None,
            ),
        ),
        graph=SimpleNamespace(name=graph_name, domain=graph_domain),
        nodes=nodes,
    )


def _write_toy_config(tmp_path: Path) -> Path:
    tmp_path.joinpath("modules_train.yaml").write_text(
        """
text:
  model:
    model_path: text
vision:
  model:
    model_path: vision
inactive:
  model:
    model_path: inactive
""",
        encoding="utf-8",
    )
    tmp_path.joinpath("graph_train.yaml").write_text(
        """
training_graph:
  - {from: vision, to: text}
  - {from: text, to: end}
""",
        encoding="utf-8",
    )
    tmp_path.joinpath("graph_infer_toy.yaml").write_text(
        """
generation_graph:
  initial: prompt
  states:
    prompt:
      body:
        - {from: vision, to: text}
        - {from: text, to: end}
      transitions:
        - {condition: {type: default}, next_state: done}
generation_kwargs:
  max_new_tokens: 1
""",
        encoding="utf-8",
    )
    return tmp_path
