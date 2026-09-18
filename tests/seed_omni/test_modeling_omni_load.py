"""Tests for HF-style OmniModel / OmniConfig checkpoint loading.

The in-tree stand-in is a two-module chain ``fake_module_a → fake_module_b``.
These tests check save layout and load; they do not feed a conversation into
``pre_forward`` / ``post_forward`` / generate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from transformers import PretrainedConfig

from veomni.models.seed_omni.configuration_omni import (
    DEFAULT_GENERATION_GRAPH_FILE,
    DEFAULT_TRAINING_GRAPH_FILE,
    OmniConfig,
)
from veomni.models.seed_omni.modeling_omni import OmniModel, merge_generation_kwargs
from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_a.modeling import FakeModuleA
from veomni.models.seed_omni.modules.fake_model.fake_module_b.configuration import FakeModuleBConfig
from veomni.models.seed_omni.modules.fake_model.fake_module_b.modeling import FakeModuleB


FAKE_A = "fake_module_a"
FAKE_B = "fake_module_b"
HIDDEN_SIZE = 8


def _chain_edges() -> list[dict]:
    return [{"from": FAKE_A, "to": FAKE_B}, {"from": FAKE_B, "to": "end"}]


def _chain_modules() -> dict:
    return {FAKE_A: {"subfolder": FAKE_A}, FAKE_B: {"subfolder": FAKE_B}}


def _write_module_stub(module_dir: Path, *, model_type: str) -> None:
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": HIDDEN_SIZE}),
        encoding="utf-8",
    )


def _write_omni_checkpoint(root: Path) -> None:
    _write_module_stub(root / FAKE_A, model_type=FAKE_A)
    _write_module_stub(root / FAKE_B, model_type=FAKE_B)

    training_graph = _chain_edges()
    generation_graphs = {
        "infer_gen": {
            "initial": "step",
            "states": {
                "step": {
                    "body": [{"from": FAKE_A, "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        },
        "infer_und": {
            "initial": "understand",
            "states": {
                "understand": {
                    "body": [{"from": FAKE_B, "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        },
    }

    yaml.safe_dump(
        {"training_graph": training_graph},
        (root / DEFAULT_TRAINING_GRAPH_FILE).open("w", encoding="utf-8"),
        sort_keys=False,
    )
    yaml.safe_dump(
        {"generation_graphs": generation_graphs},
        (root / DEFAULT_GENERATION_GRAPH_FILE).open("w", encoding="utf-8"),
        sort_keys=False,
    )

    config = {
        "model_type": "omni",
        "modules": _chain_modules(),
        "infer_type": "infer_gen",
        "generation_kwargs": {"max_new_tokens": 16},
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _minimal_generation_graph(*, module: str = FAKE_A) -> dict:
    return {
        "initial": "run",
        "states": {
            "run": {
                "body": [{"from": module, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }


def _minimal_generation_graphs(*, module: str = FAKE_A) -> dict:
    return {"infer_gen": _minimal_generation_graph(module=module)}


def _build_omni_model() -> OmniModel:
    config = OmniConfig(
        modules=_chain_modules(),
        training_graph=_chain_edges(),
        generation_graphs=_minimal_generation_graphs(),
    )
    return OmniModel(
        config,
        {
            FAKE_A: FakeModuleA(FakeModuleAConfig(hidden_size=HIDDEN_SIZE)),
            FAKE_B: FakeModuleB(FakeModuleBConfig(hidden_size=HIDDEN_SIZE)),
        },
    )


def test_omni_config_from_pretrained_hydrates_graph_sidecars(tmp_path):
    _write_omni_checkpoint(tmp_path)

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.training_graph == _chain_edges()
    assert config.infer_types == ["infer_gen", "infer_und"]
    assert config.generation_graph["initial"] == "step"
    assert config.generation_graphs["infer_und"]["initial"] == "understand"
    assert config.module_subfolder(FAKE_A) == FAKE_A
    assert config.resolve_module_path(tmp_path, FAKE_B) == str(tmp_path / FAKE_B)
    assert isinstance(config.modules[FAKE_A], FakeModuleAConfig)
    assert isinstance(config.modules[FAKE_B], FakeModuleBConfig)
    assert config.modules[FAKE_A].hidden_size == HIDDEN_SIZE


def test_omni_config_infer_type_selects_generation_graph(tmp_path):
    _write_omni_checkpoint(tmp_path)

    config = OmniConfig.from_pretrained(tmp_path)
    assert config.generation_graph["initial"] == "step"

    config.infer_type = "infer_und"
    assert config.generation_graph["initial"] == "understand"

    config.infer_type = "nope"
    with pytest.raises(KeyError, match="Unknown infer_type"):
        _ = config.generation_graph


def test_omni_config_repr_survives_required_init_args(tmp_path):
    """transformers probes `OmniConfig()` for defaults in to_diff_dict/__repr__."""
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)
    assert "omni" in repr(config)


def test_omni_config_generation_graph_is_read_only():
    config = OmniConfig(
        modules=_chain_modules(),
        training_graph=_chain_edges(),
        generation_graphs=_minimal_generation_graphs(),
    )
    with pytest.raises(AttributeError, match="read-only"):
        config.generation_graph = {"initial": "x", "states": {}}


def test_omni_config_without_scenarios_reports_clearly():
    config = OmniConfig(
        modules=_chain_modules(),
        training_graph=_chain_edges(),
        generation_graphs={},
    )
    with pytest.raises(ValueError, match="No graph scenarios"):
        _ = config.generation_graph


def test_omni_config_from_dict_rejects_legacy_generation_graph_key():
    with pytest.raises(ValueError, match="no longer a config field"):
        OmniConfig.from_dict(
            {
                "modules": _chain_modules(),
                "training_graph": _chain_edges(),
                "generation_graph": _minimal_generation_graph(),
            }
        )


def test_omni_config_rejects_legacy_single_graph_sidecar(tmp_path):
    _write_omni_checkpoint(tmp_path)
    yaml.safe_dump(
        {"generation_graph": {"initial": "step", "states": {}}},
        (tmp_path / DEFAULT_GENERATION_GRAPH_FILE).open("w", encoding="utf-8"),
        sort_keys=False,
    )

    with pytest.raises(ValueError, match="no longer supported"):
        OmniConfig.from_pretrained(tmp_path)


def test_omni_model_from_pretrained_loads_the_fake_chain(tmp_path):
    model = _build_omni_model()
    with torch.no_grad():
        model.get_module(FAKE_A).proj.weight.fill_(0.25)
        model.get_module(FAKE_B).proj.weight.fill_(0.5)
    model.save_pretrained(tmp_path)

    loaded = OmniModel.from_pretrained(tmp_path)

    assert set(loaded.modules_dict) == {FAKE_A, FAKE_B}
    assert isinstance(loaded.get_module(FAKE_A), FakeModuleA)
    assert isinstance(loaded.get_module(FAKE_B), FakeModuleB)
    assert torch.equal(loaded.get_module(FAKE_A).proj.weight, model.get_module(FAKE_A).proj.weight)
    assert torch.equal(loaded.get_module(FAKE_B).proj.weight, model.get_module(FAKE_B).proj.weight)


def test_omni_model_from_pretrained_forwards_dtype_to_modules(tmp_path):
    _build_omni_model().save_pretrained(tmp_path)

    loaded = OmniModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)

    assert loaded.get_module(FAKE_A).proj.weight.dtype == torch.bfloat16
    assert loaded.get_module(FAKE_B).proj.weight.dtype == torch.bfloat16


def test_omni_model_from_config_builds_unweighted_modules(tmp_path):
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)

    model = OmniModel.from_config(config, checkpoint_root=tmp_path)

    assert set(model.modules_dict) == {FAKE_A, FAKE_B}
    assert isinstance(model.get_module(FAKE_A), FakeModuleA)
    assert isinstance(model.get_module(FAKE_B), FakeModuleB)


def test_a_module_configured_outside_the_root_keeps_its_own_path(tmp_path):
    """Hydration must leave a module that lives elsewhere alone.

    Hydrating an entry replaces it with a ``PretrainedConfig``, and
    ``module_subfolder`` resolves those back to the bare module name. So a
    module hydrated from a configured custom path would have its weights
    looked up under ``root/<name>`` regardless — a directory that, for a module
    living outside the root, does not exist.
    """
    _write_omni_checkpoint(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    _write_module_stub(elsewhere, model_type=FAKE_A)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw["modules"][FAKE_A] = {"model": {"model_path": str(elsewhere)}}
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    shutil.rmtree(tmp_path / FAKE_A)

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.resolve_module_path(str(tmp_path), FAKE_A) == str(elsewhere)


def test_from_config_forwards_load_kwargs_to_unweighted_modules(tmp_path):
    """Building without weights must honour the load options ``__init__`` sees.

    The descriptor branch is the one that reads each module's ``config.json``
    off disk; it used to pass only the per-module ``model_config`` overrides,
    dropping the caller's ``dtype``. It must not swing the other way either: a
    global weight-placement option like ``device_map`` reaches ``__init__``
    through ``_from_config`` and would raise there.
    """
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)
    config.modules = {name: {"subfolder": name} for name in config.module_names}

    model = OmniModel.from_config(config, checkpoint_root=tmp_path, dtype="bfloat16", device_map="auto")

    assert model.get_module(FAKE_A).proj.weight.dtype == torch.bfloat16
    assert model.get_module(FAKE_B).proj.weight.dtype == torch.bfloat16


def test_from_pretrained_root_argument_wins_over_the_config_origin(tmp_path):
    """Weights come from the root the caller names, not the config's birthplace.

    ``config=`` is a documented way to load weights under an already-resolved
    config, and a config that has been through ``PreTrainedModel.from_pretrained``
    carries that call's path in ``_name_or_path``. Preferring the remembered
    path would quietly load every sub-module from the earlier checkpoint.
    """
    origin, target = tmp_path / "origin", tmp_path / "target"
    origin_model = _build_omni_model()
    with torch.no_grad():
        origin_model.get_module(FAKE_A).proj.weight.fill_(1.0)
        origin_model.get_module(FAKE_B).proj.weight.fill_(1.0)
    origin_model.save_pretrained(origin)

    target_model = _build_omni_model()
    with torch.no_grad():
        target_model.get_module(FAKE_A).proj.weight.fill_(2.0)
        target_model.get_module(FAKE_B).proj.weight.fill_(2.0)
    target_model.save_pretrained(target)

    config = OmniConfig.from_pretrained(origin)
    config.name_or_path = str(origin)

    loaded = OmniModel.from_pretrained(target, config=config)

    assert torch.equal(loaded.get_module(FAKE_A).proj.weight, target_model.get_module(FAKE_A).proj.weight)
    assert torch.equal(loaded.get_module(FAKE_B).proj.weight, target_model.get_module(FAKE_B).proj.weight)


def test_omni_config_save_pretrained_writes_graph_sidecars(tmp_path):
    config = OmniConfig(
        modules=_chain_modules(),
        training_graph=_chain_edges(),
        generation_graphs=_minimal_generation_graphs(),
    )

    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "training_graph" not in saved
    assert saved["modules"] == _chain_modules()
    assert "generation_graphs" not in saved
    assert (tmp_path / DEFAULT_GENERATION_GRAPH_FILE).exists()
    assert yaml.safe_load((tmp_path / DEFAULT_TRAINING_GRAPH_FILE).read_text(encoding="utf-8"))["training_graph"] == (
        _chain_edges()
    )
    sidecar = yaml.safe_load((tmp_path / DEFAULT_GENERATION_GRAPH_FILE).read_text(encoding="utf-8"))
    assert list(sidecar["generation_graphs"]) == ["infer_gen"]
    assert (tmp_path / "graphs" / "training.mmd").exists()
    assert (tmp_path / "graphs" / "generation_infer_gen.mmd").exists()
    training_mmd = (tmp_path / "graphs" / "training.mmd").read_text(encoding="utf-8")
    assert "flowchart" in training_mmd


def test_inference_only_config_saves_without_a_training_diagram(tmp_path):
    """No ``training_graph`` must cost the save nothing but the training diagram.

    ``from_dict`` defaults ``training_graph`` to ``[]``, which is what an
    inference-only checkpoint round-trips as, and ``TrainingGraph`` rejects an
    empty edge list. The diagrams are written last, so rendering one
    unconditionally aborted the save after every real artifact — including, via
    ``OmniModel.save_pretrained``, each module's weights — was already on disk.
    """
    config = OmniConfig.from_dict({"modules": _chain_modules(), "generation_graphs": _minimal_generation_graphs()})
    assert config.training_graph == []

    config.save_pretrained(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "graphs" / "generation_infer_gen.mmd").exists()
    assert not (tmp_path / "graphs" / "training.mmd").exists()
    assert OmniConfig.from_pretrained(tmp_path).training_graph == []


def test_omni_model_save_pretrained_roundtrip_layout(tmp_path):
    model = _build_omni_model()
    save_root = tmp_path / "saved_omni"
    model.save_pretrained(save_root, save_module_weights=False)

    assert (save_root / "config.json").exists()
    assert (save_root / DEFAULT_TRAINING_GRAPH_FILE).exists()
    assert (save_root / DEFAULT_GENERATION_GRAPH_FILE).exists()
    assert (save_root / "graphs" / "training.mmd").exists()
    assert (save_root / "graphs" / "generation_infer_gen.mmd").exists()
    assert (save_root / FAKE_A / "config.json").exists()
    assert (save_root / FAKE_B / "config.json").exists()

    reloaded = OmniConfig.from_pretrained(save_root)
    assert reloaded.training_graph[0]["from"] == FAKE_A
    assert isinstance(reloaded.modules[FAKE_A], PretrainedConfig)
    assert isinstance(reloaded.modules[FAKE_B], PretrainedConfig)
    assert json.loads((save_root / FAKE_A / "config.json").read_text(encoding="utf-8"))["model_type"] == FAKE_A
    assert json.loads((save_root / FAKE_B / "config.json").read_text(encoding="utf-8"))["model_type"] == FAKE_B


def test_save_pretrained_roundtrips_every_generation_scenario(tmp_path):
    """An exported checkpoint stays multi-scenario — it is not locked to the active one."""
    config = OmniConfig(
        modules=_chain_modules(),
        training_graph=_chain_edges(),
        generation_graphs={
            "infer_gen": _minimal_generation_graph(module=FAKE_A),
            "infer_und": {
                "initial": "understand",
                "states": {
                    "understand": {
                        "body": [{"from": FAKE_B, "to": "end"}],
                        "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                    }
                },
            },
        },
        infer_type="infer_und",
    )

    config.save_pretrained(tmp_path)
    reloaded = OmniConfig.from_pretrained(tmp_path)

    assert reloaded.infer_types == ["infer_gen", "infer_und"]
    assert reloaded.infer_type == "infer_und"
    assert reloaded.generation_graph["initial"] == "understand"
    assert reloaded.generation_graphs["infer_gen"]["initial"] == "run"
    for infer_type in reloaded.infer_types:
        assert (tmp_path / "graphs" / f"generation_{infer_type}.mmd").exists()


def test_merge_generation_kwargs_overrides_defaults():
    assert merge_generation_kwargs({"max_new_tokens": 32, "temperature": 1.0}, {"temperature": 0.5}) == {
        "max_new_tokens": 32,
        "temperature": 0.5,
    }


def test_omni_model_resolve_generation_kwargs_uses_config_defaults():
    model = _build_omni_model()
    model.config.generation_kwargs = {"max_new_tokens": 64}

    assert model.resolve_generation_kwargs(None) == {"max_new_tokens": 64}
    assert model.resolve_generation_kwargs({"temperature": 0.2}) == {
        "max_new_tokens": 64,
        "temperature": 0.2,
    }
    assert model.resolve_generation_kwargs({"max_new_tokens": 8}) == {"max_new_tokens": 8}


def test_omni_model_post_init_unions_child_no_split_modules():
    """HF ``post_init`` (re-run after children are attached) is the aggregator."""
    model = _build_omni_model()
    assert model._no_split_modules == {"FakeModuleA", "FakeModuleB"}


def test_module_checkpoint_subfolder_rejects_path_escape():
    config = OmniConfig(
        modules={"../evil": {"subfolder": "../evil"}},
        training_graph=[{"from": "../evil", "to": "end"}],
        generation_graphs=_minimal_generation_graphs(module="../evil"),
    )
    with pytest.raises(ValueError, match="not a safe checkpoint subfolder"):
        config.module_checkpoint_subfolder("../evil")
