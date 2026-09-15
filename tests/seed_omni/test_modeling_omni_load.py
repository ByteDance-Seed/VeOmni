"""Tests for HF-style OmniModel / OmniConfig checkpoint loading."""

from __future__ import annotations

import json
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
        """Mirrors ``PreTrainedModel._from_config`` — real modules have no public ``from_config``."""
        return cls(config)

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
