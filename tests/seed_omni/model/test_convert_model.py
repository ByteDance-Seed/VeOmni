"""Convert writes whatever graphs the family (or CLI) supplied; missing graphs fail at train/infer."""

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
    convert_checkpoint,
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
        "infer_und:\n"
        "  initial: run\n"
        "  states:\n"
        "    run:\n"
        "      body:\n"
        "        - {from: fake_module_a, to: end}\n"
        "      transitions:\n"
        "        - {condition: {type: default}, next_state: done}\n",
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


def test_extra_pairs_reach_the_family_converter(tmp_path):
    """``--extra KEY=VALUE`` is how a family gets an input this CLI knows nothing about.

    qwen3omni needs a second checkpoint (a Mimi encoder, since Qwen3-Omni ships
    an audio decoder but no tokenizer). Declaring that as its own CLI flag would
    put a qwen3omni-shaped argument on the entry point every other family shares,
    so the flag stays generic and the family's converter signature is what names
    the input.
    """
    from scripts.seed_omni.convert_model import _parse_extra

    assert _parse_extra(["mimi_path=kyutai/mimi", "revision=main"]) == {
        "mimi_path": "kyutai/mimi",
        "revision": "main",
    }
    # A value may itself contain '=' (query strings, base64); only the first splits.
    assert _parse_extra(["url=a=b"]) == {"url": "a=b"}
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _parse_extra(["mimi_path"])

    seen = {}
    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "extra_kwargs_test"}), encoding="utf-8")
    training_graphs, generation_graphs = load_family_graphs()

    def _record_extra(model_path: str, **kwargs) -> dict:
        del model_path
        seen.update(kwargs)
        return {
            "modules": {FAKE_A: FakeModuleA(FakeModuleAConfig()), FAKE_B: FakeModuleB(FakeModuleBConfig())},
            "training_graphs": training_graphs,
            "generation_graphs": generation_graphs,
        }

    if "extra_kwargs_test" not in OMNI_CONVERT_REGISTRY.valid_keys():
        OMNI_CONVERT_REGISTRY.register("extra_kwargs_test", lambda: _record_extra)

    convert_checkpoint(str(source), str(tmp_path / "omni"), **_parse_extra(["mimi_path=kyutai/mimi"]))
    assert seen["mimi_path"] == "kyutai/mimi"


def test_convert_fake_omni_writes_both_graphs_and_loads(tmp_path):
    source = _write_fake_omni_source(tmp_path / "src", hidden_size=8)
    output = tmp_path / "omni"
    training_graphs, generation_graphs = load_family_graphs()

    convert_checkpoint(str(source), str(output))

    assert (output / DEFAULT_TRAINING_GRAPH_FILE).is_file()
    assert (output / DEFAULT_GENERATION_GRAPH_FILE).is_file()
    config = OmniConfig.from_pretrained(output)
    assert config.training_graphs == training_graphs
    assert config.generation_graphs == generation_graphs

    loaded = OmniModel.from_pretrained(output)
    assert set(loaded.modules_dict) == {FAKE_A, FAKE_B}
    assert isinstance(loaded.get_module(FAKE_A), FakeModuleA)
    assert isinstance(loaded.get_module(FAKE_B), FakeModuleB)
    assert loaded.get_module(FAKE_A).config.hidden_size == 8


def test_convert_checkpoint_allows_omitting_graphs(tmp_path):
    """Split modules first; graphs are not required until train / generate."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "missing_graphs_test"}), encoding="utf-8")
    output = tmp_path / "omni"

    def _omit_graphs(model_path: str, **kwargs) -> dict:
        del model_path, kwargs
        return {
            "modules": {
                FAKE_A: FakeModuleA(FakeModuleAConfig()),
                FAKE_B: FakeModuleB(FakeModuleBConfig()),
            },
        }

    if "missing_graphs_test" not in OMNI_CONVERT_REGISTRY.valid_keys():
        OMNI_CONVERT_REGISTRY.register("missing_graphs_test", lambda: _omit_graphs)

    convert_checkpoint(str(source), str(output))

    assert not (output / DEFAULT_TRAINING_GRAPH_FILE).exists()
    assert not (output / DEFAULT_GENERATION_GRAPH_FILE).exists()
    config = OmniConfig.from_pretrained(output)
    assert config.training_graphs == {}
    assert config.generation_graphs == {}

    loaded = OmniModel.from_pretrained(output)
    with pytest.raises(ValueError, match="no training graph"):
        loaded({})
    with pytest.raises(ValueError, match="no generation graph"):
        loaded.generate({})


def test_convert_checkpoint_writes_only_the_graphs_it_has(tmp_path):
    """A training DAG without an FSM still converts; generate then fails until one is supplied."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "training_only_graphs_test"}), encoding="utf-8")
    output = tmp_path / "omni"
    training_graphs, _ = load_family_graphs()

    def _training_only(model_path: str, **kwargs) -> dict:
        del model_path, kwargs
        return {
            "modules": {
                FAKE_A: FakeModuleA(FakeModuleAConfig()),
                FAKE_B: FakeModuleB(FakeModuleBConfig()),
            },
            "training_graphs": training_graphs,
        }

    if "training_only_graphs_test" not in OMNI_CONVERT_REGISTRY.valid_keys():
        OMNI_CONVERT_REGISTRY.register("training_only_graphs_test", lambda: _training_only)

    convert_checkpoint(str(source), str(output))

    assert (output / DEFAULT_TRAINING_GRAPH_FILE).is_file()
    assert not (output / DEFAULT_GENERATION_GRAPH_FILE).exists()
    loaded = OmniModel.from_pretrained(output)
    assert loaded.training_graph is not None
    assert loaded.generation_graph is None
    with pytest.raises(ValueError, match="no generation graph"):
        loaded.generate({})
