"""Tests for the runtime config / OmniConfig split and per-module runtime args."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from veomni.arguments import OmniArguments, OmniDataArguments, OmniInferArguments
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.omni_arguments import OmniModelRuntimeArguments, build_module_runtime_args, build_omni_model_runtime


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Janus" / "janus_1.3b"


def _omni_args(*, model_path: str = "/tmp/janus") -> OmniArguments:
    cfg_dir = _janus_cfg_dir()
    return OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path,
            model_config={
                "modules": str(cfg_dir / "modules_train.yaml"),
                "train_graph": str(cfg_dir / "graph_train.yaml"),
                "infer_graph": {"infer_gen": str(cfg_dir / "graph_infer_gen.yaml")},
            },
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )


def _janus_model_runtime(**kwargs) -> OmniModelRuntimeArguments:
    model_path = kwargs.pop("model_path", "/tmp/janus")
    args = _omni_args(model_path=model_path)
    cfg_dir = _janus_cfg_dir()
    return build_omni_model_runtime(
        global_args=args._to_module_global_args(),
        model_path=model_path,
        train_modules=str(cfg_dir / "modules_train.yaml"),
        train_graph=kwargs.pop("train_graph", str(cfg_dir / "graph_train.yaml")),
        infer_graph=kwargs.pop("infer_graph", str(cfg_dir / "graph_infer_gen.yaml")),
        **kwargs,
    )


def test_runtime_config_keeps_the_full_launcher_view():
    """The builder returns the runtime view — nothing from the YAML is dropped."""
    runtime_cfg = _janus_model_runtime()

    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.model_path == "/tmp/janus"
    siglip = runtime_cfg.modules["janus_siglip"]
    assert siglip.accelerator.fsdp_config.fsdp_mode == "ddp"
    assert siglip.model_path.startswith("/tmp/janus")


def test_to_hf_config_projects_onto_the_checkpoint_view():
    """The HF config stores subfolder, ops_implementation, and optional model_config."""
    runtime_cfg = _janus_model_runtime()
    cfg = runtime_cfg.to_hf_config()

    assert isinstance(cfg, OmniConfig)
    assert "janus_siglip" in cfg.modules
    assert cfg.modules["janus_siglip"]["subfolder"] == "janus_siglip"
    assert "accelerator" not in cfg.modules["janus_siglip"]
    assert "optimizer" not in cfg.modules["janus_siglip"]
    assert "train" not in cfg.modules["janus_siglip"]
    assert "data" not in cfg.modules["janus_siglip"]
    # The resolved absolute `model_path` IS carried into this in-memory runtime
    # view (needed so `OmniModel.from_pretrained` / `OmniProcessor.from_config`
    # resolve a module living outside the composed checkpoint root — e.g. Qwen3
    # visual-instruction-tuning's ViT sourced from a different HF model — instead
    # of silently re-deriving the wrong `checkpoint_root/module_name` path). It
    # still never reaches an actually persisted checkpoint: `copy_for_hf_export` /
    # `normalize_modules_for_hf_export` rebuild each module's `model` block from
    # only `ops_implementation` + `model_config`, dropping `model_path`.
    assert cfg.modules["janus_siglip"]["model"]["model_path"] == runtime_cfg.modules["janus_siglip"].model_path
    for name in ("janus_vqvae", "janus_llama"):
        runtime_ops = runtime_cfg.modules[name].ops_implementation
        assert cfg.module_ops_implementation(name)["attn_implementation"] == runtime_ops.attn_implementation


def test_hf_export_strips_model_path_from_the_persisted_checkpoint():
    """`model_path` lives on the in-memory runtime view only, never on disk."""
    runtime_cfg = _janus_model_runtime()
    cfg = runtime_cfg.to_hf_config()
    assert "model_path" in cfg.modules["janus_siglip"]["model"]  # sanity: present in-memory

    exported = cfg.copy_for_hf_export()
    assert "model_path" not in exported.modules["janus_siglip"].get("model", {})


def test_to_hf_config_carries_graphs_and_scenario_selection():
    runtime_cfg = _janus_model_runtime()
    cfg = runtime_cfg.to_hf_config()

    assert cfg.training_graph == runtime_cfg.training_graph
    assert cfg.infer_types == runtime_cfg.infer_types
    assert cfg.generation_graph == runtime_cfg.generation_graph


def test_to_hf_config_does_not_alias_runtime_graphs():
    """Mutating the exported config must not reach back into the runtime config."""
    runtime_cfg = _janus_model_runtime()
    cfg = runtime_cfg.to_hf_config()

    cfg.training_graph.append({"from": "bogus", "to": "end"})
    assert {"from": "bogus", "to": "end"} not in runtime_cfg.training_graph


def test_to_hf_config_does_not_alias_nested_model_config():
    """`model_config` reaches config.json, so a nested alias could poison a checkpoint."""
    runtime_cfg = _janus_model_runtime()
    runtime_cfg.modules["janus_llama"].model_config = {"freeze": True}
    cfg = runtime_cfg.to_hf_config()

    cfg.modules["janus_llama"]["model"]["model_config"]["freeze"] = False
    assert runtime_cfg.modules["janus_llama"].model_config.get("freeze") is not False


def test_build_module_runtime_args_resolves_relative_model_paths():
    args = _omni_args(model_path="/tmp/janus")
    cfg_dir = _janus_cfg_dir()
    modules = build_module_runtime_args(
        args._to_module_global_args(),
        "/tmp/janus",
        str(cfg_dir / "modules_train.yaml"),
    )
    assert modules["janus_siglip"].model_path.startswith("/tmp/janus")


def test_build_module_runtime_args_merges_module_optimizer():
    """Global ``model.optimizer`` is the base; per-module YAML can override."""
    from veomni.arguments import OptimizerConfig
    from veomni.omni_arguments.arguments_types import OmniModuleRuntimeArguments

    global_args = OmniModuleRuntimeArguments(
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.01),
    )
    modules = build_module_runtime_args(
        global_args,
        "/tmp/janus",
        {
            "janus_llama": {"optimizer": {"lr": 2e-5, "weight_decay": 0.0}},
            "janus_siglip": {"model_path": "janus_siglip"},
        },
    )
    assert modules["janus_llama"].optimizer.lr == 2e-5
    assert modules["janus_llama"].optimizer.weight_decay == 0.0
    assert modules["janus_siglip"].optimizer.lr == 1e-4
    assert modules["janus_siglip"].optimizer.weight_decay == 0.01


def test_omni_arguments_resolve_model_modules_match_builder():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    args.model.model_config["modules"] = str(cfg_dir / "modules_infer_fsdp.yaml")
    built = args.resolve_model(for_inference=True).modules
    direct = build_module_runtime_args(
        args._to_module_global_args(),
        args.model.model_path,
        str(cfg_dir / "modules_infer_fsdp.yaml"),
        for_inference=True,
    )
    assert set(built) == set(direct)
    assert (
        built["janus_siglip"].accelerator.fsdp_config.fsdp_mode
        == direct["janus_siglip"].accelerator.fsdp_config.fsdp_mode
    )


def test_omni_arguments_resolve_model_returns_the_runtime_view():
    args = _omni_args()
    runtime_cfg = args.resolve_model()
    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.modules["janus_llama"].model_path.startswith("/tmp/janus")


def test_resolve_model_carries_every_infer_graph_scenario():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    args.model.model_config["infer_graph"] = {
        "infer_gen": str(cfg_dir / "graph_infer_gen.yaml"),
        "infer_und": str(cfg_dir / "graph_infer_und.yaml"),
    }
    args.model.set_launcher_config("infer_type", "infer_und")

    cfg = args.resolve_model()
    assert set(cfg.infer_types) == {"infer_gen", "infer_und"}
    assert cfg.infer_type == "infer_und"
    assert cfg.generation_graphs["infer_und"] is not None


def test_resolve_model_defaults_infer_type_to_first_scenario():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    args.model.model_config["infer_graph"] = {
        "infer_gen": str(cfg_dir / "graph_infer_gen.yaml"),
        "infer_und": str(cfg_dir / "graph_infer_und.yaml"),
    }
    args.model.model_config.pop("infer_type", None)

    cfg = args.resolve_model()
    assert cfg.infer_type == "infer_gen"
    assert args.model.launcher_config("infer_type") == "infer_gen"


def test_resolve_model_rejects_unknown_infer_type():
    args = _omni_args()
    args.model.set_launcher_config("infer_type", "does_not_exist")
    with pytest.raises(KeyError, match="infer_type"):
        args.resolve_model()


def test_resolve_model_carries_every_train_graph_scenario():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    args.model.model_config["train_graph"] = {
        "train": str(cfg_dir / "graph_train.yaml"),
        "alt": str(cfg_dir / "graph_train.yaml"),
    }
    args.model.set_launcher_config("train_type", "train")

    cfg = args.resolve_model()
    assert set(cfg.train_types) == {"train", "alt"}
    assert cfg.train_type == "train"


def test_resolve_model_defaults_train_type_for_single_path():
    args = _omni_args()
    args.model.model_config.pop("train_type", None)

    cfg = args.resolve_model()
    assert cfg.train_type == "default"
    assert args.model.launcher_config("train_type") == "default"


def test_resolve_model_rejects_unknown_train_type():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    args.model.model_config["train_graph"] = {
        "train": str(cfg_dir / "graph_train.yaml"),
        "alt": str(cfg_dir / "graph_train.yaml"),
    }
    args.model.set_launcher_config("train_type", "does_not_exist")
    with pytest.raises(KeyError, match="train_type"):
        args.resolve_model()


def test_infer_module_overrides_apply_eager_defaults():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    train_args = args.resolve_model().modules
    assert train_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "fsdp2"

    args.model.model_config["modules"] = str(cfg_dir / "modules_infer_eager.yaml")
    infer_args = args.resolve_model(for_inference=True).modules
    assert infer_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "eager"


def test_training_keeps_module_fsdp_modes():
    args = _omni_args()
    runtime_args = args.resolve_model().modules
    assert runtime_args["janus_siglip"].accelerator.fsdp_config.fsdp_mode == "ddp"
    assert runtime_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "fsdp2"


def test_runtime_to_hf_config_roundtrips_through_checkpoint(tmp_path):
    """Graphs / ops survive an export round-trip; module identity re-anchors under the new root.

    ``hf_cfg`` (pre-export, in-memory) carries each module's resolved absolute
    ``model_path`` (see ``test_to_hf_config_projects_onto_the_checkpoint_view``),
    so its own ``module_subfolder`` returns that absolute path. ``save_pretrained``
    strips it (``normalize_modules_for_hf_export``): the persisted config only
    ever names each module by its subfolder *relative to wherever it is
    reloaded from* — a self-contained checkpoint re-roots every module under
    itself rather than the original (possibly foreign) launcher path.
    """
    runtime_cfg = _janus_model_runtime(model_path=str(tmp_path))
    export_root = tmp_path / "exported"
    hf_cfg = runtime_cfg.to_hf_config()
    hf_cfg.save_pretrained(export_root)

    reloaded = OmniConfig.from_pretrained(export_root)
    assert reloaded.infer_types == hf_cfg.infer_types
    assert reloaded.training_graph == hf_cfg.training_graph
    assert set(reloaded.module_names) == set(hf_cfg.module_names)
    for name in hf_cfg.module_names:
        assert reloaded.module_subfolder(name) == name
        assert os.path.basename(hf_cfg.module_subfolder(name)) == name
        assert reloaded.module_ops_implementation(name) == hf_cfg.module_ops_implementation(name)


def test_resolve_model_reads_graphs_from_omni_checkpoint(tmp_path):
    """A self-contained omni checkpoint supplies graphs — no launcher YAML refs needed."""
    runtime_cfg = _janus_model_runtime(model_path=str(tmp_path))
    export_root = tmp_path / "exported"
    runtime_cfg.to_hf_config().save_pretrained(export_root)

    args = OmniArguments(
        model=OmniModelRuntimeArguments(model_path=str(export_root)),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    cfg = args.resolve_model()
    assert cfg.training_graph == runtime_cfg.training_graph
    assert cfg.infer_types == runtime_cfg.infer_types


def test_from_model_runtime_projects_onto_hf_config():
    """from_model_runtime must build OmniModel from model_runtime.to_hf_config()."""
    from unittest.mock import MagicMock, patch

    from veomni.models.seed_omni.accelerator.omni_model_runtime import OmniModelRuntime

    runtime_cfg = _janus_model_runtime()
    with patch("veomni.models.seed_omni.accelerator.module_runtime.ModuleRuntime") as mock_rt_cls:
        mock_rt_cls.return_value = MagicMock(model=MagicMock())
        with patch("veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModel") as mock_omni_model:
            OmniModelRuntime.from_model_runtime(runtime_cfg)
            omni_config = mock_omni_model.call_args[0][0]
            assert isinstance(omni_config, OmniConfig)
            assert set(omni_config.module_names) == set(runtime_cfg.module_names)
