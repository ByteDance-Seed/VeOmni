"""Convert must emit both graph sidecars for the fake_module_a → fake_module_b chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veomni.models.seed_omni.configuration_omni import (
    DEFAULT_GENERATION_GRAPH_FILE,
    DEFAULT_TRAINING_GRAPH_FILE,
    OmniConfig,
)
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.fake_model.convert_model import FAKE_A, FAKE_B, load_family_graphs
from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_a.modeling import FakeModuleA
from veomni.models.seed_omni.modules.fake_model.fake_module_b.configuration import FakeModuleBConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_b.modeling import FakeModuleB
from veomni.models.seed_omni.utils.convert_registry import (
    OMNI_CONVERT_REGISTRY,
    _require_converted_graphs,
    convert_checkpoint,
    save_converted_omni,
)


def _write_fake_omni_source(root: Path, *, hidden_size: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "fake_omni", "hidden_size": hidden_size}),
        encoding="utf-8",
    )
    return root


def test_convert_checkpoint_uses_caller_graph_yaml(tmp_path):
    """CLI-style graph paths must land in the split checkpoint, not the family default."""
    source = _write_fake_omni_source(tmp_path / "src")
    output = tmp_path / "omni"
    train_yaml = tmp_path / "train.yaml"
    infer_yaml = tmp_path / "infer.yaml"
    train_yaml.write_text(
        "- {from: fake_module_a, to: end}\n",
        encoding="utf-8",
    )
    infer_yaml.write_text(
        "generation_graphs:\n"
        "  infer_und:\n"
        "    initial: run\n"
        "    states:\n"
        "      run:\n"
        "        body:\n"
        "          - {from: fake_module_a, to: end}\n"
        "        transitions:\n"
        "          - {condition: {type: default}, next_state: done}\n",
        encoding="utf-8",
    )

    convert_checkpoint(
        str(source),
        str(output),
        training_graph=str(train_yaml),
        generation_graph=str(infer_yaml),
    )

    config = OmniConfig.from_pretrained(output)
    assert config.training_graph == [{"from": "fake_module_a", "to": "end"}]
    assert list(config.generation_graphs) == ["infer_und"]
    assert config.infer_type == "infer_und"
    assert config.generation_graph["initial"] == "run"


def test_convert_fake_omni_writes_both_graphs_and_loads(tmp_path):
    source = _write_fake_omni_source(tmp_path / "src", hidden_size=8)
    output = tmp_path / "omni"
    training_graph, generation_graphs = load_family_graphs()

    convert_checkpoint(str(source), str(output))

    assert (output / DEFAULT_TRAINING_GRAPH_FILE).is_file()
    assert (output / DEFAULT_GENERATION_GRAPH_FILE).is_file()
    config = OmniConfig.from_pretrained(output)
    assert config.training_graph == training_graph
    assert config.generation_graphs == generation_graphs

    loaded = OmniModel.from_pretrained(output)
    assert set(loaded.modules_dict) == {FAKE_A, FAKE_B}
    assert isinstance(loaded.get_module(FAKE_A), FakeModuleA)
    assert isinstance(loaded.get_module(FAKE_B), FakeModuleB)
    assert loaded.get_module(FAKE_A).config.hidden_size == 8


def test_convert_checkpoint_rejects_missing_generation_graph(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "missing_graphs_test"}), encoding="utf-8")
    output = tmp_path / "omni"
    training_graph, _ = load_family_graphs()

    def _omit_generation_graphs(model_path: str, **kwargs) -> dict:
        del model_path, kwargs
        return {
            "modules": {
                FAKE_A: FakeModuleA(FakeModuleAConfig()),
                FAKE_B: FakeModuleB(FakeModuleBConfig()),
            },
            "training_graph": training_graph,
            "generation_graphs": {},
        }

    if "missing_graphs_test" not in OMNI_CONVERT_REGISTRY.valid_keys():
        OMNI_CONVERT_REGISTRY.register("missing_graphs_test", lambda: _omit_generation_graphs)

    with pytest.raises(ValueError, match="generation-graph"):
        convert_checkpoint(str(source), str(output))


def test_require_converted_graphs_rejects_empty_training_graph(tmp_path):
    training_graph, generation_graphs = load_family_graphs()
    save_converted_omni(
        str(tmp_path),
        modules={
            FAKE_A: FakeModuleA(FakeModuleAConfig()),
            FAKE_B: FakeModuleB(FakeModuleBConfig()),
        },
        training_graph=training_graph,
        generation_graphs=generation_graphs,
    )
    (tmp_path / DEFAULT_TRAINING_GRAPH_FILE).write_text("training_graph: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        _require_converted_graphs(str(tmp_path))
