"""Tests for shared graph-driven V2 model loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.seed_omni.parity_suite.core import NodeSpec
from tests.seed_omni.parity_suite.core.discovery import discover_cases
from tests.seed_omni.parity_suite.v2.model import graph_active_module_names, load_graph_active_omni_config


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


def test_bagel_inference_uses_full_selected_graph_module_set() -> None:
    [case] = [
        case
        for case in discover_cases([Path("tests/seed_omni/bagel")])
        if case.recipe.id == "text_und" and case.tier == "graph" and case.run.id == "one_step"
    ]

    module_names = graph_active_module_names(case)
    config = load_graph_active_omni_config(case)

    assert "bagel_siglip_navit" in module_names
    assert module_names == frozenset(config.module_names)
    assert config.training_graph == []


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
