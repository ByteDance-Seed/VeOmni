"""Tests for the runtime config / OmniConfig split and per-module runtime args."""

from __future__ import annotations

import json
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


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Janus" / "janus_1.3b"


def _omni_args(*, model_path: str = "/tmp/janus") -> OmniArguments:
    cfg_dir = _janus_cfg_dir()
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


def _janus_model_runtime(**kwargs) -> OmniModelRuntimeArguments:
    model_path = kwargs.pop("model_path", "/tmp/janus")
    args = _omni_args(model_path=model_path)
    cfg_dir = _janus_cfg_dir()
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
    runtime_cfg = _janus_model_runtime()

    assert isinstance(runtime_cfg, OmniModelRuntimeArguments)
    assert runtime_cfg.model_path == "/tmp/janus"
    siglip = runtime_cfg.modules["janus_siglip"]
    assert siglip.accelerator.fsdp_config.fsdp_mode == "ddp"
    assert siglip.model_path.startswith("/tmp/janus")


def test_to_hf_config_projects_onto_the_checkpoint_view():
    """The HF config stores subfolder, this module's kernels, and optional model_config."""
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
    # `normalize_modules_for_hf_export` rebuild each module from subfolder +
    # optional `model_config` / `ops_implementation`, dropping `model_path`.
    assert cfg.modules["janus_siglip"]["model"]["model_path"] == runtime_cfg.modules["janus_siglip"].model_path
    # Ops DO travel with the module. Per-module kernel choices are set per module
    # in the launcher YAML (`janus_vqvae` runs eager attention where the LLM does
    # not), and the native load path — `OmniModel.from_pretrained` with no
    # launcher — has no other carrier for them: without this, every slot on every
    # module falls back to the caller's single global config, which for a MoE
    # module means the eager per-expert dispatch.
    for name in ("janus_vqvae", "janus_llama"):
        runtime_ops = runtime_cfg.modules[name].ops_implementation
        assert runtime_ops.attn_implementation is not None
        model_block = cfg.modules[name].get("model") or {}
        assert model_block["ops_implementation"]["attn_implementation"] == runtime_ops.attn_implementation


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

    assert cfg.training_graphs == runtime_cfg.training_graphs
    assert cfg.train_type == runtime_cfg.train_type
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
        str(cfg_dir / "train/modules_train.yaml"),
    )
    assert modules["janus_siglip"].model_path.startswith("/tmp/janus")


def test_packed_modules_yaml_sets_text_encoder_processor_config():
    args = _omni_args(model_path="/tmp/janus")
    modules = build_module_runtime_args(
        args._to_module_global_args(),
        "/tmp/janus",
        str(_janus_cfg_dir() / "packed/modules_train.yaml"),
    )
    encoder = modules["janus_text_encoder"]
    assert encoder.processor_config == {"packed_preprocess": True}
    assert not encoder.model_config.get("packed_preprocess")

    entry = encoder.to_hf_config("janus_text_encoder")
    cfg = OmniConfig(
        modules={"janus_text_encoder": entry},
        training_graphs={"default": [{"from": "janus_text_encoder", "to": "end"}]},
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )
    assert cfg.module_processor_config("janus_text_encoder") == {"packed_preprocess": True}
    exported = cfg.copy_for_hf_export()
    assert exported.modules["janus_text_encoder"]["processor_config"] == {"packed_preprocess": True}


def test_build_module_runtime_args_merges_module_optimizer():
    """Global ``model.optimizer`` is the base; per-module YAML can override."""
    from veomni.arguments import OptimizerConfig
    from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments

    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/janus",
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
    args.model.model_config["modules"] = str(cfg_dir / "infer/modules_infer_fsdp.yaml")
    built = args.resolve_model(for_inference=True).modules
    direct = build_module_runtime_args(
        args._to_module_global_args(),
        args.model.model_path,
        str(cfg_dir / "infer/modules_infer_fsdp.yaml"),
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
    cfg_dir = _janus_cfg_dir()
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
    cfg_dir = _janus_cfg_dir()
    args.model.model_config["train_graph"] = {
        "train": str(cfg_dir / "train/graph_train.yaml"),
        "alt": str(cfg_dir / "train/graph_train.yaml"),
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
        "train": str(cfg_dir / "train/graph_train.yaml"),
        "alt": str(cfg_dir / "train/graph_train.yaml"),
    }
    args.model.set_launcher_config("train_type", "does_not_exist")
    with pytest.raises(KeyError, match="train_type"):
        args.resolve_model()


def test_infer_module_overrides_apply_eager_defaults():
    args = _omni_args()
    cfg_dir = _janus_cfg_dir()
    train_args = args.resolve_model().modules
    assert train_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "fsdp2"

    args.model.model_config["modules"] = str(cfg_dir / "infer/modules_infer_eager.yaml")
    infer_args = args.resolve_model(for_inference=True).modules
    assert infer_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "eager"


def test_training_keeps_module_fsdp_modes():
    args = _omni_args()
    runtime_args = args.resolve_model().modules
    assert runtime_args["janus_siglip"].accelerator.fsdp_config.fsdp_mode == "ddp"
    assert runtime_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "fsdp2"


def test_runtime_to_hf_config_roundtrips_through_checkpoint(tmp_path):
    """Graphs survive an export round-trip; module identity re-anchors under the new root.

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
        exported_model = reloaded.normalize_modules_for_hf_export()[name].get("model") or {}
        assert "model_path" not in exported_model
        # Kernels survive load→save. `from_pretrained` replaces each descriptor
        # with a hydrated family config, which has nowhere to keep the `model`
        # block, so without the stash this is where a re-export would drop them.
        assert (
            exported_model["ops_implementation"]["attn_implementation"]
            == runtime_cfg.modules[name].ops_implementation.attn_implementation
        )


def test_launcher_module_args_win_over_the_persisted_kernels(tmp_path):
    """Persisting kernels does not freeze them into the checkpoint.

    The checkpoint's value is the fallback for a caller that has no launcher —
    a bare ``OmniModel.from_pretrained(root)``. Everything that *does* go
    through VeOmni re-states it from the per-module launcher args instead:
    ``ModuleRuntime._build_model`` passes ``args.ops_implementation`` to
    ``build_foundation_model`` without consulting :class:`OmniConfig` at all,
    and the eager-inference path (``OmniInferencer``) hands
    ``OmniModel.from_pretrained`` a ``config=`` projected from those same args.
    This pins the second one, which is the one that reads the config and so the
    one that could plausibly have picked the stale value up.
    """
    export_root = tmp_path / "exported"
    _janus_model_runtime(model_path=str(tmp_path)).to_hf_config().save_pretrained(export_root)

    persisted = OmniConfig.from_pretrained(export_root)
    assert persisted.module_ops_implementation("janus_llama")["attn_implementation"] is not None

    relaunched = _janus_model_runtime(model_path=str(tmp_path))
    relaunched.modules["janus_llama"].ops_implementation.attn_implementation = "sdpa"

    # What OmniInferencer passes as `config=` on the eager path.
    from_launcher = relaunched.to_hf_config()
    assert from_launcher.module_ops_implementation("janus_llama")["attn_implementation"] == "sdpa"


def _exported_janus_root(tmp_path) -> Path:
    exported = tmp_path / "exported"
    _janus_model_runtime(model_path=str(tmp_path)).to_hf_config().save_pretrained(exported)
    return exported


def test_the_checkpoint_fills_in_what_the_launcher_yaml_leaves_out(tmp_path):
    """A launcher YAML no longer shuts the checkpoint's own values out.

    These used to be either/or: name a ``modules:`` YAML and the checkpoint's
    entries were never consulted. That made the persisted kernels reachable
    only by a launcher-less caller, even though the checkpoint is the one layer
    that knows a module individually without the user restating it. Janus shows
    the cost — ``modules_train.yaml`` pins each module's attention, and no infer
    YAML repeats it, so an inference run silently dropped to the global value.
    """
    from veomni.arguments import OpsImplementationConfig

    exported = _exported_janus_root(tmp_path)
    # `janus_vqvae`, because `modules_train.yaml` pins it to eager attention
    # while the global default is the SP-aware flash variant. Asserting on a
    # module whose exported value happens to match the global would pass whether
    # or not the checkpoint was consulted at all.
    trained_attn = OmniConfig.from_pretrained(exported).module_ops_implementation("janus_vqvae")["attn_implementation"]
    assert trained_attn != OpsImplementationConfig().attn_implementation

    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=str(exported),
            model_config={"modules": str(_janus_cfg_dir() / "infer/modules_infer_fsdp.yaml")},
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    modules = args.resolve_model(for_inference=True).modules

    # The infer YAML names no kernels, so the exported ones stand...
    assert modules["janus_vqvae"].ops_implementation.attn_implementation == trained_attn
    # ...and it does name parallelism, which the checkpoint must not undo.
    assert modules["janus_siglip"].accelerator.fsdp_config.fsdp_mode == "eager"


def test_a_launcher_yaml_still_wins_where_it_names_the_same_field(tmp_path):
    """The checkpoint is a default, not a freeze.

    Swapping a training kernel for an inference one is the whole reason the
    checkpoint layer sits *under* the YAML rather than over it. Deep-merged, so
    naming one kernel does not drop the module's others.
    """
    exported = _exported_janus_root(tmp_path)
    modules_yaml = {
        name: {"ops_implementation": {"attn_implementation": "sdpa"}} if name == "janus_llama" else {}
        for name in OmniConfig.from_pretrained(exported).module_names
    }
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=str(exported),
            model_config={"modules": modules_yaml},
        ),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    modules = args.resolve_model().modules

    assert modules["janus_llama"].ops_implementation.attn_implementation == "sdpa"
    # Untouched by the YAML, so still the exported value rather than the global.
    vqvae_attn = OmniConfig.from_pretrained(exported).module_ops_implementation("janus_vqvae")["attn_implementation"]
    assert modules["janus_vqvae"].ops_implementation.attn_implementation == vqvae_attn


def _janus_args_over(exported: Path, modules: dict) -> OmniArguments:
    return OmniArguments(
        model=OmniModelRuntimeArguments(model_path=str(exported), model_config={"modules": modules}),
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
        payload["modules"][name]["accelerator"] = {"fsdp_config": {"fsdp_mode": mode}}
    config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_a_checkpoint_accelerator_overlay_reaches_training(tmp_path):
    """Parallelism layers like everything else, and the YAML still wins."""
    exported = _exported_janus_root(tmp_path)
    _with_checkpoint_accelerator(exported, janus_vqvae="ddp", janus_siglip="ddp")

    modules = (
        _janus_args_over(
            exported,
            {"janus_vqvae": {}, "janus_siglip": {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}}}},
        )
        .resolve_model()
        .modules
    )

    assert modules["janus_vqvae"].accelerator.fsdp_config.fsdp_mode == "ddp"
    assert modules["janus_siglip"].accelerator.fsdp_config.fsdp_mode == "fsdp2"


def test_inference_stays_eager_over_a_checkpoint_accelerator(tmp_path):
    """A checkpoint must not make an inference run distributed on its own.

    Parallelism belongs to a run, not a checkpoint — the reason `to_hf_config`
    declines to persist `accelerator` in the first place. So the synthesized
    inference default sits *above* the checkpoint layer: `eager` is what an
    inference run gets unless its own YAML says otherwise, and a checkpoint
    `accelerator` that would silently pull in FSDP2 collectives is masked.
    """
    exported = _exported_janus_root(tmp_path)
    _with_checkpoint_accelerator(exported, janus_vqvae="fsdp2", janus_siglip="fsdp2")

    modules = (
        _janus_args_over(
            exported,
            {"janus_vqvae": {}, "janus_siglip": {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}}}},
        )
        .resolve_model(for_inference=True)
        .modules
    )

    # `janus_vqvae: {}` names the module without saying anything about it, so the
    # layers beneath it stand.
    assert modules["janus_vqvae"].accelerator.fsdp_config.fsdp_mode == "eager"
    # Still overridable by the run's own YAML, which is the layer above.
    assert modules["janus_siglip"].accelerator.fsdp_config.fsdp_mode == "ddp"


def test_a_launcher_less_inference_run_is_eager_and_keeps_the_exported_kernels(tmp_path):
    """No ``modules:`` YAML at all: the checkpoint decides the module set.

    The layer that names the modules is also the one with nothing to say about
    them, so this is where a layering slip shows up first — every module's entry
    is synthesized empty and the eager default has only the checkpoint beneath it.
    """
    exported = _exported_janus_root(tmp_path)
    _with_checkpoint_accelerator(exported, janus_vqvae="fsdp2")
    trained_attn = OmniConfig.from_pretrained(exported).module_ops_implementation("janus_vqvae")["attn_implementation"]

    args = OmniArguments(
        model=OmniModelRuntimeArguments(model_path=str(exported)),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )
    modules = args.resolve_model(for_inference=True).modules

    assert set(modules) == set(OmniConfig.from_pretrained(exported).module_names)
    assert modules["janus_vqvae"].accelerator.fsdp_config.fsdp_mode == "eager"
    assert not modules["janus_vqvae"].broadcast_model_weights_from_rank0
    # The eager default masks parallelism only — the exported kernels still land.
    assert modules["janus_vqvae"].ops_implementation.attn_implementation == trained_attn
    assert Path(modules["janus_vqvae"].model_path) == exported / "janus_vqvae"


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

    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    runtime_cfg = _janus_model_runtime()
    with patch("veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime") as mock_rt_cls:
        mock_rt_cls.return_value = MagicMock(model=MagicMock())
        with patch("veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.OmniModel") as mock_omni_model:
            OmniModelRuntime.from_model_runtime(runtime_cfg)
            omni_config = mock_omni_model.call_args[0][0]
            assert isinstance(omni_config, OmniConfig)
            assert set(omni_config.module_names) == set(runtime_cfg.module_names)


def test_omni_module_config_owns_descriptor_conversion():
    """Per-module path / export logic lives on OmniModuleConfig, not OmniConfig."""
    from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig

    entry = OmniModuleConfig.from_runtime(
        "janus_siglip",
        model_path="/tmp/janus/janus_siglip",
        model_config={"freeze": True},
        processor_config={"packed_preprocess": True},
    )
    cfg = OmniModuleConfig("janus_siglip", entry)
    assert cfg.subfolder == "/tmp/janus/janus_siglip"
    assert cfg.checkpoint_subfolder == "janus_siglip"
    assert cfg.model_config_overrides() == {"freeze": True}
    assert cfg.processor_config() == {"packed_preprocess": True}
    exported = cfg.to_export_dict()
    assert exported == {
        "subfolder": "janus_siglip",
        "model": {"model_config": {"freeze": True}},
        "processor_config": {"packed_preprocess": True},
    }
    assert "model_path" not in exported.get("model", {})


def test_omni_module_runtime_config_is_the_arguments_alias():
    from veomni.arguments import OmniModelRuntimeArguments, OmniModuleRuntimeArguments
    from veomni.models.seed_omni.accelerated import OmniModelRuntimeConfig, OmniModuleRuntimeConfig

    assert OmniModuleRuntimeArguments is OmniModuleRuntimeConfig
    assert OmniModelRuntimeArguments is OmniModelRuntimeConfig
