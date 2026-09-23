"""Tests for the runtime config / OmniConfig split and per-module runtime args.

Driven by the in-tree ``fake_module_a -> fake_module_b`` chain
(``configs/seed_omni/fake_model/``) so argument resolution is covered without
depending on any real model's configs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from veomni.arguments import (
    OmniArguments,
    OmniDataArguments,
    OmniInferArguments,
    OmniModelRuntimeArguments,
    build_module_runtime_args,
    build_omni_model_runtime,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig


MODULE_A = "fake_module_a"
MODULE_B = "fake_module_b"


def _cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "fake_model"


def _omni_args(*, model_path: str = "/tmp/fake_omni") -> OmniArguments:
    cfg_dir = _cfg_dir()
    return OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path,
            model_config={
                "modules": str(cfg_dir / "train/modules_train.yaml"),
                "train_graph": str(cfg_dir / "train/graph_train.yaml"),
                "infer_graph": {"infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml")},
            },
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )


def _with_module_configs(cfg: OmniConfig) -> OmniConfig:
    """Attach the module configs an ``OmniModel`` would have taken off live modules.

    ``OmniConfig.save_pretrained`` writes each module's own ``config.json`` into
    its subfolder, and only ``OmniModel.__init__`` (from the modules it is given)
    or ``OmniConfig.from_pretrained`` (from disk) fill ``_module_configs``. A
    config projected straight from launcher arguments has never seen a module,
    so a test that saves one has to stand in for that step.
    """
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
    from veomni.models.seed_omni.modules.fake_model.fake_module_b.configuration import FakeModuleBConfig

    config_classes = {MODULE_A: FakeModuleAConfig, MODULE_B: FakeModuleBConfig}
    for name in cfg.module_names:
        cfg._module_configs[name] = config_classes[name]()
    return cfg


def _model_runtime(**kwargs) -> OmniModelRuntimeArguments:
    model_path = kwargs.pop("model_path", "/tmp/fake_omni")
    args = _omni_args(model_path=model_path)
    cfg_dir = _cfg_dir()
    return build_omni_model_runtime(
        global_args=args._to_module_global_args(),
        model_path=model_path,
        train_modules=str(cfg_dir / "train/modules_train.yaml"),
        train_graph=kwargs.pop("train_graph", str(cfg_dir / "train/graph_train.yaml")),
        infer_graph=kwargs.pop("infer_graph", str(cfg_dir / "infer/graph_infer_gen.yaml")),
        **kwargs,
    )


def test_runtime_config_keeps_the_full_launcher_view():
    """The builder returns the runtime view — nothing from the YAML is dropped."""
    runtime_cfg = _model_runtime()

    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.model_path == "/tmp/fake_omni"
    module_a = runtime_cfg.modules[MODULE_A]
    assert module_a.accelerator.fsdp_config.fsdp_mode == "ddp"
    assert module_a.model_path.startswith("/tmp/fake_omni")


def test_to_hf_config_projects_onto_the_checkpoint_view():
    """Each module becomes one flat checkpoint entry; ops stay on the runtime args."""
    runtime_cfg = _model_runtime()
    cfg = runtime_cfg.to_hf_config()

    assert isinstance(cfg, OmniConfig)
    assert set(cfg.module_names) == {MODULE_A, MODULE_B}
    entry = cfg._module_entries[MODULE_A]
    for launcher_only in ("accelerator", "optimizer", "train", "data"):
        assert launcher_only not in entry
    # The resolved absolute `model_path` IS carried into this in-memory runtime
    # view (needed so `OmniModel.from_pretrained` / `OmniProcessor.from_config`
    # resolve a module living outside the composed checkpoint root — e.g. a ViT
    # sourced from a different HF model — instead of silently re-deriving the
    # wrong `checkpoint_root/module_name` path). It still never reaches an
    # actually persisted checkpoint: `OmniConfig.to_dict` renames every entry
    # back to its own subfolder. Ops live on the runtime args and are never
    # written onto OmniConfig.
    assert entry["model_path"] == runtime_cfg.modules[MODULE_A].model_path
    for name in (MODULE_A, MODULE_B):
        runtime_ops = runtime_cfg.modules[name].ops_implementation
        assert runtime_ops.attn_implementation is not None
        assert "ops_implementation" not in cfg._module_entries[name]


def test_hf_export_strips_model_path_from_the_persisted_checkpoint():
    """`model_path` is an in-memory load path; a persisted entry names its own subfolder."""
    runtime_cfg = _model_runtime()
    cfg = runtime_cfg.to_hf_config()
    # sanity: in-memory it is wherever the launcher resolved the module to
    assert cfg._module_entries[MODULE_A]["model_path"] != MODULE_A

    exported = cfg.copy_for_hf_export()
    assert exported._module_entries[MODULE_A]["model_path"] == MODULE_A


def test_to_hf_config_carries_graphs_and_scenario_selection():
    runtime_cfg = _model_runtime()
    cfg = runtime_cfg.to_hf_config()

    assert cfg.training_graph == runtime_cfg.training_graph
    assert cfg.infer_types == runtime_cfg.infer_types
    assert cfg.generation_graph == runtime_cfg.generation_graph


def test_to_hf_config_does_not_alias_runtime_graphs():
    """Mutating the exported config must not reach back into the runtime config."""
    runtime_cfg = _model_runtime()
    cfg = runtime_cfg.to_hf_config()

    cfg.training_graph.append({"from": "bogus", "to": "end"})
    assert {"from": "bogus", "to": "end"} not in runtime_cfg.training_graph


def test_to_hf_config_does_not_alias_nested_model_config():
    """`model_config` reaches config.json, so a nested alias could poison a checkpoint."""
    runtime_cfg = _model_runtime()
    runtime_cfg.modules[MODULE_B].model_config = {"freeze": True}
    cfg = runtime_cfg.to_hf_config()

    cfg._module_entries[MODULE_B]["model_config"]["freeze"] = False
    assert runtime_cfg.modules[MODULE_B].model_config.get("freeze") is not False


def test_build_module_runtime_args_resolves_relative_model_paths():
    args = _omni_args(model_path="/tmp/fake_omni")
    modules = build_module_runtime_args(
        args._to_module_global_args(),
        "/tmp/fake_omni",
        str(_cfg_dir() / "train/modules_train.yaml"),
    )
    assert modules[MODULE_A].model_path.startswith("/tmp/fake_omni")


def test_packed_modules_yaml_sets_module_processor_config():
    args = _omni_args(model_path="/tmp/fake_omni")
    modules = build_module_runtime_args(
        args._to_module_global_args(),
        "/tmp/fake_omni",
        str(_cfg_dir() / "train/modules_train_packed.yaml"),
    )
    module_a = modules[MODULE_A]
    assert module_a.processor_config == {"packed_preprocess": True}
    assert not module_a.model_config.get("packed_preprocess")

    cfg = OmniConfig(
        _module_entries={MODULE_A: module_a.to_hf_config()},
        training_graphs={"default": [{"from": MODULE_A, "to": "end"}]},
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )
    assert cfg._module_entries[MODULE_A]["processor_config"] == {"packed_preprocess": True}
    exported = cfg.copy_for_hf_export()
    assert exported._module_entries[MODULE_A]["processor_config"] == {"packed_preprocess": True}


def test_build_module_runtime_args_merges_module_optimizer():
    """Global ``model.optimizer`` is the base; per-module YAML can override."""
    from veomni.arguments import OptimizerConfig
    from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments

    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/fake_omni",
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.01),
    )
    modules = build_module_runtime_args(
        global_args,
        "/tmp/fake_omni",
        {
            MODULE_B: {"optimizer": {"lr": 2e-5, "weight_decay": 0.0}},
            MODULE_A: {"model_path": MODULE_A},
        },
    )
    assert modules[MODULE_B].optimizer.lr == 2e-5
    assert modules[MODULE_B].optimizer.weight_decay == 0.0
    assert modules[MODULE_A].optimizer.lr == 1e-4
    assert modules[MODULE_A].optimizer.weight_decay == 0.01


def test_omni_arguments_resolve_model_modules_match_builder():
    args = _omni_args()
    modules_yaml = str(_cfg_dir() / "infer/modules_infer_fsdp.yaml")
    args.model.model_config["modules"] = modules_yaml
    built = args.resolve_model(for_inference=True).modules
    direct = build_module_runtime_args(
        args._to_module_global_args(),
        args.model.model_path,
        modules_yaml,
        for_inference=True,
    )
    assert set(built) == set(direct)
    assert built[MODULE_A].accelerator.fsdp_config.fsdp_mode == direct[MODULE_A].accelerator.fsdp_config.fsdp_mode


def test_omni_arguments_resolve_model_returns_the_runtime_view():
    args = _omni_args()
    runtime_cfg = args.resolve_model()
    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.modules[MODULE_B].model_path.startswith("/tmp/fake_omni")


def test_resolve_model_carries_every_infer_graph_scenario():
    args = _omni_args()
    cfg_dir = _cfg_dir()
    args.model.model_config["infer_graph"] = {
        "infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml"),
        "infer_und": str(cfg_dir / "infer/graph_infer_und.yaml"),
    }
    args.model.set_launcher_config("infer_type", "infer_und")

    cfg = args.resolve_model()
    assert set(cfg.infer_types) == {"infer_gen", "infer_und"}
    assert cfg.infer_type == "infer_und"
    assert cfg.generation_graphs["infer_und"] is not None


def test_resolve_model_defaults_infer_type_to_first_scenario():
    args = _omni_args()
    cfg_dir = _cfg_dir()
    args.model.model_config["infer_graph"] = {
        "infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml"),
        "infer_und": str(cfg_dir / "infer/graph_infer_und.yaml"),
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
    train_graph = str(_cfg_dir() / "train/graph_train.yaml")
    args.model.model_config["train_graph"] = {"train": train_graph, "alt": train_graph}
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
    train_graph = str(_cfg_dir() / "train/graph_train.yaml")
    args.model.model_config["train_graph"] = {"train": train_graph, "alt": train_graph}
    args.model.set_launcher_config("train_type", "does_not_exist")
    with pytest.raises(KeyError, match="train_type"):
        args.resolve_model()


def test_infer_module_overrides_apply_eager_defaults():
    args = _omni_args()
    train_args = args.resolve_model().modules
    assert train_args[MODULE_B].accelerator.fsdp_config.fsdp_mode == "fsdp2"

    args.model.model_config["modules"] = str(_cfg_dir() / "infer/modules_infer_eager.yaml")
    infer_args = args.resolve_model(for_inference=True).modules
    assert infer_args[MODULE_B].accelerator.fsdp_config.fsdp_mode == "eager"


def test_training_keeps_module_fsdp_modes():
    args = _omni_args()
    runtime_args = args.resolve_model().modules
    assert runtime_args[MODULE_A].accelerator.fsdp_config.fsdp_mode == "ddp"
    assert runtime_args[MODULE_B].accelerator.fsdp_config.fsdp_mode == "fsdp2"


def test_runtime_to_hf_config_roundtrips_through_checkpoint(tmp_path):
    """Graphs survive an export round-trip; module identity re-anchors under the new root.

    ``hf_cfg`` (pre-export, in-memory) carries each module's resolved absolute
    ``model_path`` (see ``test_to_hf_config_projects_onto_the_checkpoint_view``).
    Saving re-roots them: the persisted config only ever names each module by
    its subfolder *relative to wherever it is reloaded from*, so a
    self-contained checkpoint re-anchors every module under itself rather than
    the original (possibly foreign) launcher path.
    """
    runtime_cfg = _model_runtime(model_path=str(tmp_path))
    export_root = tmp_path / "exported"
    hf_cfg = _with_module_configs(runtime_cfg.to_hf_config())
    # Read before saving: `to_dict` rewrites these onto the module names.
    loaded_from = {name: hf_cfg._module_entries[name]["model_path"] for name in hf_cfg.module_names}
    hf_cfg.save_pretrained(export_root)

    reloaded = OmniConfig.from_pretrained(export_root)
    assert reloaded.infer_types == hf_cfg.infer_types
    assert reloaded.training_graph == hf_cfg.training_graph
    assert set(reloaded.module_names) == set(hf_cfg.module_names)
    for name in hf_cfg.module_names:
        assert reloaded._module_entries[name]["model_path"] == name
        assert os.path.basename(loaded_from[name]) == name
        assert not reloaded._module_entries[name].get("ops_implementation")


def test_resolve_model_reads_graphs_from_omni_checkpoint(tmp_path):
    """A self-contained omni checkpoint supplies graphs — no launcher YAML refs needed."""
    runtime_cfg = _model_runtime(model_path=str(tmp_path))
    export_root = tmp_path / "exported"
    _with_module_configs(runtime_cfg.to_hf_config()).save_pretrained(export_root)

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

    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    runtime_cfg = _model_runtime()
    with patch("veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime") as mock_rt_cls:
        mock_rt_cls.return_value = MagicMock(model=MagicMock())
        with patch("veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.OmniModel") as mock_omni_model:
            OmniModelRuntime.from_model_runtime(runtime_cfg)
            omni_config = mock_omni_model.call_args[0][0]
            assert isinstance(omni_config, OmniConfig)
            assert set(omni_config.module_names) == set(runtime_cfg.module_names)


def test_omni_module_config_owns_descriptor_conversion():
    """Per-module path / export logic lives on OmniModuleConfig, not OmniConfig."""
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
    from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig

    # An entry pointing outside the composed root keeps its own path; one
    # without a path is the module's subfolder under that root.
    external = f"/tmp/elsewhere/{MODULE_A}"
    assert OmniModuleConfig.resolve_path("/tmp/fake_omni", MODULE_A, external) == external
    assert OmniModuleConfig.resolve_path("/tmp/fake_omni", MODULE_A, None) == f"/tmp/fake_omni/{MODULE_A}"

    cfg = FakeModuleAConfig()
    cfg._apply_composed_overwrites(
        model_config={"freeze": True},
        processor_config={"packed_preprocess": True},
        ops_implementation=None,
        base_ops_implementation=None,
    )
    assert cfg.freeze is True
    assert cfg.processor_config == {"packed_preprocess": True}

    # A module's own config.json states what the module is, not where the
    # composed model loaded it from nor the blocks it never set.
    exported = cfg.to_diff_dict()
    assert exported["processor_config"] == {"packed_preprocess": True}
    assert "model_path" not in exported
    assert "ops_implementation" not in exported


def test_omni_module_runtime_config_is_the_arguments_alias():
    from veomni.arguments import OmniModelRuntimeArguments, OmniModuleRuntimeArguments
    from veomni.models.seed_omni.accelerated import OmniModelRuntimeConfig, OmniModuleRuntimeConfig

    assert OmniModuleRuntimeArguments is OmniModuleRuntimeConfig
    assert OmniModelRuntimeArguments is OmniModelRuntimeConfig
