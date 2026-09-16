"""Tests for HF-style OmniModel / OmniConfig checkpoint loading."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
import yaml
from transformers import PretrainedConfig

from veomni.models.seed_omni.configuration_omni import (
    DEFAULT_GENERATION_GRAPH_FILE,
    DEFAULT_TRAINING_GRAPH_FILE,
    OmniConfig,
)
from veomni.models.seed_omni.modeling_omni import OmniModel, merge_generation_kwargs
from veomni.models.seed_omni.utils.visualize import (
    GRAPH_VIS_SUBDIR,
    TRAINING_MMD_FILENAME,
    generation_mmd_filename,
)


def _write_module_stub(module_dir: Path, *, model_type: str = "fake_omni_module") -> None:
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 4}),
        encoding="utf-8",
    )


def _write_omni_checkpoint(root: Path) -> None:
    _write_module_stub(root / "encoder")
    _write_module_stub(root / "decoder")

    training_graph = [{"from": "encoder", "to": "decoder"}, {"from": "decoder", "to": "end"}]
    generation_graphs = {
        "infer_gen": {
            "initial": "step",
            "states": {
                "step": {
                    "body": [{"from": "encoder", "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        },
        "infer_und": {
            "initial": "understand",
            "states": {
                "understand": {
                    "body": [{"from": "decoder", "to": "end"}],
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
        "modules": {
            "encoder": {"subfolder": "encoder"},
            "decoder": {"subfolder": "decoder"},
        },
        "infer_type": "infer_gen",
        "generation_kwargs": {"max_new_tokens": 16},
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


class _FakeModuleConfig(PretrainedConfig):
    model_type = "fake_omni_module"


class _FakeModule(nn.Module):
    config_class = _FakeModuleConfig

    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def _from_config(cls, config, **kwargs):
        """Mirrors ``PreTrainedModel._from_config`` — real modules have no public ``from_config``.

        Including the part that matters here: it pops the handful of options it
        understands and hands everything else to ``__init__``, so a caller that
        forwards a weight-loading option gets a ``TypeError``.
        """
        captured = getattr(cls, "_captured_from_config_kwargs", [])
        captured.append(dict(kwargs))
        cls._captured_from_config_kwargs = captured
        for key in ("dtype", "torch_dtype", "attn_implementation", "experts_implementation"):
            kwargs.pop(key, None)
        return cls(config, **kwargs)

    @classmethod
    def from_pretrained(cls, module_path, **kwargs):
        captured = getattr(cls, "_captured_kwargs", {})
        captured[str(module_path)] = dict(kwargs)
        cls._captured_kwargs = captured
        cfg = _FakeModuleConfig.from_pretrained(module_path)
        return cls(cfg)

    def save_pretrained(self, save_directory, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)


def test_omni_config_from_pretrained_hydrates_graph_sidecars(tmp_path):
    _write_omni_checkpoint(tmp_path)

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.training_graph == [
        {"from": "encoder", "to": "decoder"},
        {"from": "decoder", "to": "end"},
    ]
    assert config.infer_types == ["infer_gen", "infer_und"]
    assert config.generation_graph["initial"] == "step"
    assert config.generation_graphs["infer_und"]["initial"] == "understand"
    assert config.module_subfolder("encoder") == "encoder"
    assert config.resolve_module_path(tmp_path, "decoder") == str(tmp_path / "decoder")
    assert isinstance(config.modules["encoder"], PretrainedConfig)
    assert isinstance(config.modules["decoder"], PretrainedConfig)


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
        modules={"encoder": {"subfolder": "encoder"}},
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs=_minimal_generation_graphs(),
    )
    with pytest.raises(AttributeError, match="read-only"):
        config.generation_graph = {"initial": "x", "states": {}}


def test_omni_config_without_scenarios_reports_clearly():
    config = OmniConfig(
        modules={"encoder": {"subfolder": "encoder"}},
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs={},
    )
    with pytest.raises(ValueError, match="No graph scenarios"):
        _ = config.generation_graph


def test_omni_config_from_dict_rejects_legacy_generation_graph_key():
    with pytest.raises(ValueError, match="no longer a config field"):
        OmniConfig.from_dict(
            {
                "modules": {"encoder": {"subfolder": "encoder"}},
                "training_graph": [{"from": "encoder", "to": "end"}],
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


@patch("veomni.models.seed_omni.modeling_omni.read_model_type", return_value="fake_omni_module")
@patch("veomni.models.seed_omni.modeling_omni.OMNI_MODEL_REGISTRY")
def test_omni_model_from_pretrained_forwards_kwargs_to_modules(registry_mock, _read_model_type, tmp_path):
    _write_omni_checkpoint(tmp_path)

    fake_cls = _FakeModule
    fake_cls._captured_kwargs = {}
    registry_mock.__getitem__.return_value = MagicMock(return_value=fake_cls)

    model = OmniModel.from_pretrained(tmp_path, torch_dtype="bfloat16", device_map="auto")

    assert isinstance(model, OmniModel)
    assert set(model.modules_dict) == {"encoder", "decoder"}
    assert fake_cls._captured_kwargs[str(tmp_path / "encoder")]["torch_dtype"] == "bfloat16"
    assert fake_cls._captured_kwargs[str(tmp_path / "decoder")]["device_map"] == "auto"


@patch("veomni.models.seed_omni.modeling_omni.read_model_type", return_value="fake_omni_module")
@patch("veomni.models.seed_omni.modeling_omni.OMNI_MODEL_REGISTRY")
def test_omni_model_from_config_builds_unweighted_modules(registry_mock, _read_model_type, tmp_path):
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)

    fake_cls = _FakeModule
    registry_mock.__getitem__.return_value = MagicMock(return_value=fake_cls)

    model = OmniModel.from_config(config, checkpoint_root=tmp_path)

    assert set(model.modules_dict) == {"encoder", "decoder"}


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
    _write_module_stub(elsewhere)
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text())
    raw["modules"]["encoder"] = {"model": {"model_path": str(elsewhere)}}
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    shutil.rmtree(tmp_path / "encoder")

    config = OmniConfig.from_pretrained(tmp_path)

    assert config.resolve_module_path(str(tmp_path), "encoder") == str(elsewhere)


@patch("veomni.models.seed_omni.modeling_omni.read_model_type", return_value="fake_omni_module")
@patch("veomni.models.seed_omni.modeling_omni.OMNI_MODEL_REGISTRY")
def test_from_config_forwards_load_kwargs_to_unweighted_modules(registry_mock, _read_model_type, tmp_path):
    """Building without weights must honour the load options ``__init__`` sees.

    The descriptor branch is the one that reads each module's ``config.json``
    off disk; it used to pass only the per-module ``model_config`` overrides,
    dropping the caller's ``dtype`` and any ``attn_implementation`` persisted
    in ``ops_implementation``. It must not swing the other way either: a global
    weight-placement option like ``device_map`` reaches ``__init__`` through
    ``_from_config`` and would raise there.
    """
    _write_omni_checkpoint(tmp_path)
    config = OmniConfig.from_pretrained(tmp_path)
    config.modules = {name: {"subfolder": name} for name in config.module_names}  # keep the descriptor branch

    fake_cls = _FakeModule
    fake_cls._captured_from_config_kwargs = []
    registry_mock.__getitem__.return_value = MagicMock(return_value=fake_cls)

    OmniModel.from_config(config, checkpoint_root=tmp_path, dtype="bfloat16", device_map="auto")

    assert fake_cls._captured_from_config_kwargs
    assert all(captured == {"dtype": "bfloat16"} for captured in fake_cls._captured_from_config_kwargs)


@patch("veomni.models.seed_omni.modeling_omni.read_model_type", return_value="fake_omni_module")
@patch("veomni.models.seed_omni.modeling_omni.OMNI_MODEL_REGISTRY")
def test_from_pretrained_root_argument_wins_over_the_config_origin(registry_mock, _read_model_type, tmp_path):
    """Weights come from the root the caller names, not the config's birthplace.

    ``config=`` is a documented way to load weights under an already-resolved
    config, and a config that has been through ``PreTrainedModel.from_pretrained``
    carries that call's path in ``_name_or_path``. Preferring the remembered
    path would quietly load every sub-module from the earlier checkpoint.
    """
    origin, target = tmp_path / "origin", tmp_path / "target"
    _write_omni_checkpoint(origin)
    _write_omni_checkpoint(target)

    config = OmniConfig.from_pretrained(origin)
    config.name_or_path = str(origin)  # what transformers stamps on after a model load

    fake_cls = _FakeModule
    fake_cls._captured_kwargs = {}
    registry_mock.__getitem__.return_value = MagicMock(return_value=fake_cls)

    OmniModel.from_pretrained(target, config=config)

    assert set(fake_cls._captured_kwargs) == {str(target / "encoder"), str(target / "decoder")}


def _minimal_generation_graph(*, module: str = "encoder") -> dict:
    return {
        "initial": "run",
        "states": {
            "run": {
                "body": [{"from": module, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }


def _minimal_generation_graphs(*, module: str = "encoder") -> dict:
    return {"infer_gen": _minimal_generation_graph(module=module)}


def test_omni_config_save_pretrained_writes_graph_sidecars(tmp_path):
    config = OmniConfig(
        modules={"encoder": {"subfolder": "encoder"}},
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs=_minimal_generation_graphs(),
    )

    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "training_graph" not in saved
    assert saved["modules"] == {"encoder": {"subfolder": "encoder"}}
    assert "generation_graphs" not in saved
    assert (tmp_path / DEFAULT_GENERATION_GRAPH_FILE).exists()
    assert yaml.safe_load((tmp_path / DEFAULT_TRAINING_GRAPH_FILE).read_text(encoding="utf-8"))["training_graph"] == [
        {"from": "encoder", "to": "end"}
    ]
    sidecar = yaml.safe_load((tmp_path / DEFAULT_GENERATION_GRAPH_FILE).read_text(encoding="utf-8"))
    assert list(sidecar["generation_graphs"]) == ["infer_gen"]
    assert (tmp_path / GRAPH_VIS_SUBDIR / TRAINING_MMD_FILENAME).exists()
    assert (tmp_path / GRAPH_VIS_SUBDIR / generation_mmd_filename("infer_gen")).exists()
    training_mmd = (tmp_path / GRAPH_VIS_SUBDIR / TRAINING_MMD_FILENAME).read_text(encoding="utf-8")
    assert "flowchart" in training_mmd


def test_inference_only_config_saves_without_a_training_diagram(tmp_path):
    """No ``training_graph`` must cost the save nothing but the training diagram.

    ``from_dict`` defaults ``training_graph`` to ``[]``, which is what an
    inference-only checkpoint round-trips as, and ``TrainingGraph`` rejects an
    empty edge list. The diagrams are written last, so rendering one
    unconditionally aborted the save after every real artifact — including, via
    ``OmniModel.save_pretrained``, each module's weights — was already on disk.
    """
    config = OmniConfig.from_dict(
        {"modules": {"encoder": {"subfolder": "encoder"}}, "generation_graphs": _minimal_generation_graphs()}
    )
    assert config.training_graph == []

    config.save_pretrained(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / GRAPH_VIS_SUBDIR / generation_mmd_filename("infer_gen")).exists()
    assert not (tmp_path / GRAPH_VIS_SUBDIR / TRAINING_MMD_FILENAME).exists()
    assert OmniConfig.from_pretrained(tmp_path).training_graph == []


@patch("veomni.models.seed_omni.modeling_omni.read_model_type", return_value="fake_omni_module")
@patch("veomni.models.seed_omni.modeling_omni.OMNI_MODEL_REGISTRY")
def test_omni_model_save_pretrained_roundtrip_layout(registry_mock, _read_model_type, tmp_path):
    fake_cls = _FakeModule
    registry_mock.__getitem__.return_value = MagicMock(return_value=fake_cls)

    config = OmniConfig(
        modules={
            "encoder": {"subfolder": "encoder"},
            "decoder": {"subfolder": "decoder"},
        },
        training_graph=[{"from": "encoder", "to": "decoder"}, {"from": "decoder", "to": "end"}],
        generation_graphs=_minimal_generation_graphs(module="encoder"),
    )
    modules = {
        "encoder": fake_cls(_FakeModuleConfig(hidden_size=4)),
        "decoder": fake_cls(_FakeModuleConfig(hidden_size=4)),
    }
    model = OmniModel(config, modules)

    save_root = tmp_path / "saved_omni"
    model.save_pretrained(save_root, save_module_weights=False)

    assert (save_root / "config.json").exists()
    assert (save_root / DEFAULT_TRAINING_GRAPH_FILE).exists()
    assert (save_root / DEFAULT_GENERATION_GRAPH_FILE).exists()
    assert (save_root / GRAPH_VIS_SUBDIR / TRAINING_MMD_FILENAME).exists()
    assert (save_root / GRAPH_VIS_SUBDIR / generation_mmd_filename("infer_gen")).exists()
    assert (save_root / "encoder" / "config.json").exists()
    assert (save_root / "decoder" / "config.json").exists()

    reloaded = OmniConfig.from_pretrained(save_root)
    assert reloaded.training_graph[0]["from"] == "encoder"
    from transformers import PretrainedConfig

    assert isinstance(reloaded.modules["encoder"], PretrainedConfig)
    assert isinstance(reloaded.modules["decoder"], PretrainedConfig)


def test_save_pretrained_roundtrips_every_generation_scenario(tmp_path):
    """An exported checkpoint stays multi-scenario — it is not locked to the active one."""
    config = OmniConfig(
        modules={"encoder": {"subfolder": "encoder"}},
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs={
            "infer_gen": _minimal_generation_graph(module="encoder"),
            "infer_und": {
                "initial": "understand",
                "states": {
                    "understand": {
                        "body": [{"from": "encoder", "to": "end"}],
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
        assert (tmp_path / GRAPH_VIS_SUBDIR / generation_mmd_filename(infer_type)).exists()


def test_merge_generation_kwargs_overrides_defaults():
    assert merge_generation_kwargs({"max_new_tokens": 32, "temperature": 1.0}, {"temperature": 0.5}) == {
        "max_new_tokens": 32,
        "temperature": 0.5,
    }


def test_omni_model_resolve_generation_kwargs_uses_config_defaults():
    config = OmniConfig(
        modules={"encoder": {"subfolder": "encoder"}},
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs=_minimal_generation_graphs(),
        generation_kwargs={"max_new_tokens": 64},
    )
    model = OmniModel(config, {"encoder": _FakeModule(_FakeModuleConfig(hidden_size=4))})

    assert model.resolve_generation_kwargs(None) == {"max_new_tokens": 64}
    assert model.resolve_generation_kwargs({"temperature": 0.2}) == {
        "max_new_tokens": 64,
        "temperature": 0.2,
    }
    assert model.resolve_generation_kwargs({"max_new_tokens": 8}) == {"max_new_tokens": 8}


def test_omni_model_post_init_unions_child_no_split_modules():
    """HF ``post_init`` (re-run after children are attached) is the aggregator."""

    class _Encoder(nn.Module):
        _no_split_modules = ["EncoderLayer"]

        def __init__(self):
            super().__init__()

    class _Decoder(nn.Module):
        _no_split_modules = ["DecoderLayer", "EncoderLayer"]

        def __init__(self):
            super().__init__()

    config = OmniConfig(
        modules={"encoder": {"subfolder": "encoder"}, "decoder": {"subfolder": "decoder"}},
        training_graph=[{"from": "encoder", "to": "decoder"}, {"from": "decoder", "to": "end"}],
        generation_graphs=_minimal_generation_graphs(),
    )
    model = OmniModel(config, {"encoder": _Encoder(), "decoder": _Decoder()})
    assert model._no_split_modules == {"EncoderLayer", "DecoderLayer"}
