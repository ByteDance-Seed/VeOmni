"""Tests for the runtime config / OmniConfig split and per-module runtime args.

Driven by the in-tree ``fake_module_a -> fake_module_b`` chain
(``configs/seed_omni/fake_model/``) so argument resolution is covered without
depending on any real model's configs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from veomni.arguments.omni_arguments_types import (
    OmniArguments,
    OmniDataArguments,
    OmniInferArguments,
    OmniModelRuntimeArguments,
    build_omni_model_runtime_args,
    build_omni_module_runtime_args,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig


MODULE_A = "fake_module_a"
MODULE_B = "fake_module_b"


def _cfg_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "seed_omni" / "fake_model"


def _omni_args(*, model_path: str = "/tmp/fake_omni", **launcher_config) -> OmniArguments:
    cfg_dir = _cfg_dir()
    return OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path,
            model_config={
                "modules": str(cfg_dir / "train/modules_train.yaml"),
                "train_graph": str(cfg_dir / "train/graph_train.yaml"),
                "infer_graph": {"infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml")},
                **launcher_config,
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
    return build_omni_model_runtime_args(_omni_args(**kwargs))


def test_runtime_config_keeps_the_full_launcher_view():
    """The builder returns the runtime view — nothing from the YAML is dropped."""
    runtime_cfg = _model_runtime()

    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.model_path == "/tmp/fake_omni"
    module_a = runtime_cfg.modules[MODULE_A]
    assert module_a.accelerator.fsdp_config.fsdp_mode == "ddp"
    assert module_a.model_path.startswith("/tmp/fake_omni")


def test_to_hf_config_projects_onto_the_checkpoint_view():
    """Each module becomes one flat checkpoint entry, kernels included."""
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
    # back to its own subfolder. Kernels are carried so the checkpoint
    # remembers what each module was trained with.
    assert entry["model_path"] == runtime_cfg.modules[MODULE_A].model_path
    for name in (MODULE_A, MODULE_B):
        runtime_ops = runtime_cfg.modules[name].ops_implementation
        assert runtime_ops.attn_implementation is not None
        assert cfg._module_entries[name]["ops_implementation"]["attn_implementation"] == (
            runtime_ops.attn_implementation
        )


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


def test_build_omni_module_runtime_args_resolves_relative_model_paths():
    args = _omni_args(model_path="/tmp/fake_omni")
    modules = build_omni_module_runtime_args(
        args._to_module_global_args(),
        "/tmp/fake_omni",
        str(_cfg_dir() / "train/modules_train.yaml"),
    )
    assert modules[MODULE_A].model_path.startswith("/tmp/fake_omni")


def test_build_omni_module_runtime_args_merges_module_optimizer():
    """Global ``model.optimizer`` is the base; per-module YAML can override."""
    from veomni.arguments import OptimizerConfig
    from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments

    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/fake_omni",
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.01),
    )
    modules = build_omni_module_runtime_args(
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


def test_omni_arguments_build_model_runtime_args_modules_match_builder():
    args = _omni_args()
    modules_yaml = str(_cfg_dir() / "infer/modules_infer_fsdp.yaml")
    args.model.model_config["modules"] = modules_yaml
    built = build_omni_model_runtime_args(args, for_inference=True).modules
    direct = build_omni_module_runtime_args(
        args._to_module_global_args(),
        args.model.model_path,
        modules_yaml,
        for_inference=True,
    )
    assert set(built) == set(direct)
    assert built[MODULE_A].accelerator.fsdp_config.fsdp_mode == direct[MODULE_A].accelerator.fsdp_config.fsdp_mode


def test_omni_arguments_build_model_runtime_args_returns_the_runtime_view():
    args = _omni_args()
    runtime_cfg = build_omni_model_runtime_args(args)
    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.modules[MODULE_B].model_path.startswith("/tmp/fake_omni")


def test_build_model_runtime_args_carries_every_infer_graph_scenario():
    args = _omni_args()
    cfg_dir = _cfg_dir()
    args.model.model_config["infer_graph"] = {
        "infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml"),
        "infer_und": str(cfg_dir / "infer/graph_infer_und.yaml"),
    }
    args.model.set_launcher_config("infer_type", "infer_und")

    cfg = build_omni_model_runtime_args(args)
    assert set(cfg.infer_types) == {"infer_gen", "infer_und"}
    assert cfg.infer_type == "infer_und"
    assert cfg.generation_graphs["infer_und"] is not None


def test_build_model_runtime_args_defaults_infer_type_to_first_scenario():
    args = _omni_args()
    cfg_dir = _cfg_dir()
    args.model.model_config["infer_graph"] = {
        "infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml"),
        "infer_und": str(cfg_dir / "infer/graph_infer_und.yaml"),
    }
    args.model.model_config.pop("infer_type", None)

    cfg = build_omni_model_runtime_args(args)
    assert cfg.infer_type == "infer_gen"
    assert args.model.launcher_config("infer_type") == "infer_gen"


def test_build_model_runtime_args_rejects_unknown_infer_type():
    args = _omni_args()
    args.model.set_launcher_config("infer_type", "does_not_exist")
    with pytest.raises(KeyError, match="infer_type"):
        build_omni_model_runtime_args(args)


def test_build_model_runtime_args_carries_every_train_graph_scenario():
    args = _omni_args()
    train_graph = str(_cfg_dir() / "train/graph_train.yaml")
    args.model.model_config["train_graph"] = {"train": train_graph, "alt": train_graph}
    args.model.set_launcher_config("train_type", "train")

    cfg = build_omni_model_runtime_args(args)
    assert set(cfg.train_types) == {"train", "alt"}
    assert cfg.train_type == "train"


def test_build_model_runtime_args_defaults_train_type_for_single_path():
    args = _omni_args()
    args.model.model_config.pop("train_type", None)

    cfg = build_omni_model_runtime_args(args)
    assert cfg.train_type == "default"
    assert args.model.launcher_config("train_type") == "default"


def test_build_model_runtime_args_rejects_unknown_train_type():
    args = _omni_args()
    train_graph = str(_cfg_dir() / "train/graph_train.yaml")
    args.model.model_config["train_graph"] = {"train": train_graph, "alt": train_graph}
    args.model.set_launcher_config("train_type", "does_not_exist")
    with pytest.raises(KeyError, match="train_type"):
        build_omni_model_runtime_args(args)


def test_infer_module_overrides_apply_eager_defaults():
    args = _omni_args()
    train_args = build_omni_model_runtime_args(args).modules
    assert train_args[MODULE_B].accelerator.fsdp_config.fsdp_mode == "fsdp2"

    args.model.model_config["modules"] = str(_cfg_dir() / "infer/modules_infer_eager.yaml")
    infer_args = build_omni_model_runtime_args(args, for_inference=True).modules
    assert infer_args[MODULE_B].accelerator.fsdp_config.fsdp_mode == "eager"


def test_a_model_scope_wrap_rejects_eager_modules_before_loading_them():
    """Inference defaults modules to eager but not the top level, so a
    ``fsdp_scope='model'`` run would otherwise load all weights, then fail at wrap time."""
    args = _omni_args()
    args.model.accelerator.fsdp_config.fsdp_scope = "model"
    assert build_omni_model_runtime_args(args).modules[MODULE_B].accelerator.fsdp_config.fsdp_mode == "fsdp2"

    args.model.model_config["modules"] = str(_cfg_dir() / "infer/modules_infer_eager.yaml")
    with pytest.raises(ValueError, match=rf"fsdp_mode='eager': \['{MODULE_A}', '{MODULE_B}'\]"):
        build_omni_model_runtime_args(args, for_inference=True)


def test_training_keeps_module_fsdp_modes():
    args = _omni_args()
    runtime_args = build_omni_model_runtime_args(args).modules
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
        assert (
            reloaded._module_entries[name]["ops_implementation"]["attn_implementation"]
            == runtime_cfg.modules[name].ops_implementation.attn_implementation
        )


def _exported_root(tmp_path) -> Path:
    exported = tmp_path / "exported"
    _with_module_configs(_model_runtime(model_path=str(tmp_path)).to_hf_config()).save_pretrained(exported)
    return exported


def _exported_attn(exported: Path, name: str) -> str:
    return OmniConfig.from_pretrained(exported)._module_entries[name]["ops_implementation"]["attn_implementation"]


def _args_over(exported: Path, modules: dict | str | None = None) -> OmniArguments:
    model_config = {} if modules is None else {"modules": modules}
    return OmniArguments(
        model=OmniModelRuntimeArguments(model_path=str(exported), model_config=model_config),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )


def _with_checkpoint_accelerator(exported: Path, **by_module: str) -> None:
    """Hand-write an ``accelerator`` onto checkpoint entries.

    Export never writes one — ``to_hf_config`` projects the model fields only —
    so a hand-written ``config.json`` is the only way this layer carries one.
    """
    config_file = exported / "config.json"
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    for name, mode in by_module.items():
        payload["_module_entries"][name]["accelerator"] = {"fsdp_config": {"fsdp_mode": mode}}
    config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_launcher_module_args_win_over_the_persisted_kernels(tmp_path):
    """Persisting kernels does not freeze them into the checkpoint.

    The eager-inference path (``OmniInferencer``) hands
    ``OmniModel.from_pretrained`` a ``config=`` projected from the launcher's
    per-module args, so a relaunch that changes a kernel must see its own value.
    """
    exported = _exported_root(tmp_path)
    assert _exported_attn(exported, MODULE_B) not in (None, "sdpa")

    relaunched = _model_runtime(model_path=str(tmp_path))
    relaunched.modules[MODULE_B].ops_implementation.attn_implementation = "sdpa"

    from_launcher = relaunched.to_hf_config()
    assert from_launcher._module_entries[MODULE_B]["ops_implementation"]["attn_implementation"] == "sdpa"


def test_the_checkpoint_fills_in_what_the_launcher_yaml_leaves_out(tmp_path):
    """A launcher YAML does not shut the checkpoint's own values out.

    ``modules_train.yaml`` pins each module's attention and no infer YAML repeats
    it, so without the checkpoint layer an inference run would silently drop to
    the global value.
    """
    from veomni.arguments import OpsImplementationConfig

    exported = _exported_root(tmp_path)
    # `fake_module_a` is pinned to eager attention, unlike the global default:
    # a module whose exported value matched the global would pass whether or
    # not the checkpoint was consulted.
    trained_attn = _exported_attn(exported, MODULE_A)
    assert trained_attn != OpsImplementationConfig().attn_implementation

    modules = build_omni_model_runtime_args(
        _args_over(exported, str(_cfg_dir() / "infer/modules_infer_eager.yaml")), for_inference=True
    ).modules

    # The infer YAML names no kernels, so the exported ones stand...
    assert modules[MODULE_A].ops_implementation.attn_implementation == trained_attn
    # ...and it does name parallelism, which the checkpoint must not undo.
    assert modules[MODULE_A].accelerator.fsdp_config.fsdp_mode == "eager"


def test_a_launcher_yaml_still_wins_where_it_names_the_same_field(tmp_path):
    """The checkpoint is a default, not a freeze; deep-merged per field."""
    exported = _exported_root(tmp_path)
    modules_yaml = {MODULE_A: {}, MODULE_B: {"ops_implementation": {"attn_implementation": "sdpa"}}}

    modules = build_omni_model_runtime_args(_args_over(exported, modules_yaml)).modules

    assert modules[MODULE_B].ops_implementation.attn_implementation == "sdpa"
    # Untouched by the YAML, so still the exported value rather than the global.
    assert modules[MODULE_A].ops_implementation.attn_implementation == _exported_attn(exported, MODULE_A)


def test_a_checkpoint_accelerator_overlay_reaches_training(tmp_path):
    """Parallelism layers like everything else, and the YAML still wins."""
    exported = _exported_root(tmp_path)
    _with_checkpoint_accelerator(exported, **{MODULE_A: "ddp", MODULE_B: "ddp"})

    modules = build_omni_model_runtime_args(
        _args_over(exported, {MODULE_A: {}, MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}}}})
    ).modules

    assert modules[MODULE_A].accelerator.fsdp_config.fsdp_mode == "ddp"
    assert modules[MODULE_B].accelerator.fsdp_config.fsdp_mode == "fsdp2"


def test_inference_stays_eager_over_a_checkpoint_accelerator(tmp_path):
    """A checkpoint must not make an inference run distributed on its own."""
    exported = _exported_root(tmp_path)
    _with_checkpoint_accelerator(exported, **{MODULE_A: "fsdp2", MODULE_B: "fsdp2"})

    modules = build_omni_model_runtime_args(
        _args_over(exported, {MODULE_A: {}, MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}}}}),
        for_inference=True,
    ).modules

    # `{}` names the module without saying anything about it, so the layers beneath it stand.
    assert modules[MODULE_A].accelerator.fsdp_config.fsdp_mode == "eager"
    # Still overridable by the run's own YAML, which is the layer above.
    assert modules[MODULE_B].accelerator.fsdp_config.fsdp_mode == "ddp"


def test_a_launcher_less_inference_run_is_eager_and_keeps_the_exported_kernels(tmp_path):
    """No ``modules:`` YAML at all: the checkpoint decides the module set."""
    exported = _exported_root(tmp_path)
    _with_checkpoint_accelerator(exported, **{MODULE_A: "fsdp2"})
    trained_attn = _exported_attn(exported, MODULE_A)

    modules = build_omni_model_runtime_args(_args_over(exported), for_inference=True).modules

    assert set(modules) == {MODULE_A, MODULE_B}
    assert modules[MODULE_A].accelerator.fsdp_config.fsdp_mode == "eager"
    assert not modules[MODULE_A].broadcast_model_weights_from_rank0
    # The eager default masks parallelism only — the exported kernels still land.
    assert modules[MODULE_A].ops_implementation.attn_implementation == trained_attn
    assert Path(modules[MODULE_A].model_path) == exported / MODULE_A


def test_build_model_runtime_args_reads_graphs_from_omni_checkpoint(tmp_path):
    """A self-contained omni checkpoint supplies graphs — no launcher YAML refs needed."""
    runtime_cfg = _model_runtime(model_path=str(tmp_path))
    export_root = tmp_path / "exported"
    _with_module_configs(runtime_cfg.to_hf_config()).save_pretrained(export_root)

    args = OmniArguments(
        model=OmniModelRuntimeArguments(model_path=str(export_root)),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    cfg = build_omni_model_runtime_args(args)
    assert cfg.training_graph == runtime_cfg.training_graph
    assert cfg.infer_types == runtime_cfg.infer_types


def test_a_launcher_infer_graph_does_not_inherit_the_checkpoint_infer_type(tmp_path):
    """The checkpoint's ``infer_type`` names one of its own scenarios, not the launcher's."""
    infer_dir = _cfg_dir() / "infer"
    runtime_cfg = _model_runtime(
        model_path=str(tmp_path),
        infer_graph={"infer_und": str(infer_dir / "graph_infer_und.yaml")},
        infer_type="infer_und",
    )
    export_root = tmp_path / "exported"
    _with_module_configs(runtime_cfg.to_hf_config()).save_pretrained(export_root)
    assert OmniConfig.from_pretrained(export_root).infer_type == "infer_und"

    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=str(export_root),
            model_config={"infer_graph": {"understanding": str(infer_dir / "graph_infer_und.yaml")}},
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    assert build_omni_model_runtime_args(args, for_inference=True).infer_type == "understanding"


def test_build_model_runtime_args_keeps_every_training_scenario_from_omni_checkpoint(tmp_path):
    """The checkpoint stores the whole ``training_graphs`` map and its ``train_type``;
    reading back only the active graph would rename it ``default`` and drop the rest."""
    graph = str(_cfg_dir() / "train/graph_train.yaml")
    runtime_cfg = _model_runtime(
        model_path=str(tmp_path), train_graph={"train": graph, "alt": graph}, train_type="alt"
    )
    export_root = tmp_path / "exported"
    _with_module_configs(runtime_cfg.to_hf_config()).save_pretrained(export_root)

    args = OmniArguments(
        model=OmniModelRuntimeArguments(model_path=str(export_root)),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    cfg = build_omni_model_runtime_args(args)
    assert list(cfg.training_graphs) == ["train", "alt"]
    assert cfg.train_type == "alt"


def test_build_omni_model_runtime_projects_onto_hf_config(tmp_path):
    """build_omni_model_runtime builds OmniModel from ``to_hf_config()``; each ModuleRuntime
    gets the module config that OmniConfig loaded, and OmniModel gets the bare module."""
    from unittest.mock import MagicMock, patch

    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import build_omni_model_runtime

    for name, config in _with_module_configs(_model_runtime().to_hf_config())._module_configs.items():
        config.save_pretrained(tmp_path / name)
    runtime_cfg = _model_runtime(model_path=str(tmp_path))
    with patch("veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime") as mock_rt_cls:
        mock_rt_cls.side_effect = lambda *a, **k: MagicMock(model=MagicMock(), omni_module=MagicMock())
        with patch("veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.OmniModel") as mock_omni_model:
            runtime = build_omni_model_runtime(runtime_cfg)
            omni_config, modules = mock_omni_model.call_args[0]
            assert isinstance(omni_config, OmniConfig)
            assert set(omni_config.module_names) == set(runtime_cfg.module_names)

    for call in mock_rt_cls.call_args_list:
        name = call.kwargs["module_name"]
        assert call.kwargs["module_config"] is omni_config._module_configs[name]
    for name, module_runtime in runtime.module_runtimes.items():
        assert modules[name] is module_runtime.omni_module


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
        processor_config={"image_size": 224},
        ops_implementation=None,
        base_ops_implementation=None,
    )
    assert cfg.freeze is True
    assert cfg.processor_config == {"image_size": 224}

    # A module's own config.json states what the module is, not where the
    # composed model loaded it from nor the blocks it never set.
    exported = cfg.to_diff_dict()
    assert exported["processor_config"] == {"image_size": 224}
    assert "model_path" not in exported
    assert "ops_implementation" not in exported
