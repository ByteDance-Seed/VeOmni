"""Tests for the per-module accelerator training knobs moved off `OmniTrainingArguments`.

Covers: `AcceleratorConfig` gaining the SeedOmni-only fields, per-module
`accelerator.*` override survival through `build_omni_module_runtime_args`, and
`_validate_omni_accelerator` being invoked both for the top-level default
(`OmniArguments.__post_init__`) and per resolved module (`build_omni_model_runtime_args`).

The resolution tests run on the in-tree ``fake_module_a -> fake_module_b``
chain (``configs/seed_omni/fake_model/``), so they need no real model config.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import (
    AcceleratorConfig,
    FSDPConfig,
    GradientCheckpointingConfig,
    TorchCompileConfig,
)
from veomni.arguments.omni_arguments_types import (
    OmniArguments,
    OmniDataArguments,
    OmniInferArguments,
    OmniModelRuntimeArguments,
    OmniModuleRuntimeArguments,
    OmniTrainingArguments,
    _validate_omni_accelerator,
    build_omni_model_runtime_args,
    build_omni_module_runtime_args,
)
from veomni.models.model_runtime import VeOmniModelRuntime


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


MODULE_A = "fake_module_a"
MODULE_B = "fake_module_b"


def _cfg_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "seed_omni" / "fake_model"


def _fake_args(*, modules_override: dict | None = None, accelerator: AcceleratorConfig | None = None) -> OmniArguments:
    cfg_dir = _cfg_dir()
    modules_yaml = str(cfg_dir / "train/modules_train.yaml")
    model_config = {
        "modules": modules_yaml,
        "train_graph": str(cfg_dir / "train/graph_train.yaml"),
        "infer_graph": {"infer_gen": str(cfg_dir / "infer/graph_infer_gen.yaml")},
    }
    if modules_override is not None:
        with open(modules_yaml) as f:
            loaded = yaml.safe_load(f)
        for name, override in modules_override.items():
            loaded.setdefault(name, {})
            loaded[name] = {**loaded[name], **override}
        model_config["modules"] = loaded
    model_kwargs = {"model_path": "/tmp/fake_omni", "model_config": model_config}
    if accelerator is not None:
        model_kwargs["accelerator"] = accelerator
    return OmniArguments(
        model=OmniModelRuntimeArguments(**model_kwargs),
        data=OmniDataArguments(train_path=""),
        infer=OmniInferArguments(),
    )


def test_accelerator_config_has_seed_omni_fields_with_expected_defaults():
    acc = AcceleratorConfig()
    assert acc.init_device == "meta"
    assert isinstance(acc.gradient_checkpointing, GradientCheckpointingConfig)
    assert isinstance(acc.torch_compile, TorchCompileConfig)
    assert not hasattr(acc, "broadcast_model_weights_from_rank0")
    assert not hasattr(acc, "ep_sharded_stream_load")


def test_fsdp_scope_defaults_to_module_and_rejects_unknown_values():
    assert AcceleratorConfig().fsdp_config.fsdp_scope == "module"
    with pytest.raises(ValueError, match="fsdp_scope"):
        FSDPConfig(fsdp_scope="everything")


def test_the_generic_runtime_refuses_eager_rather_than_falling_back_to_ddp():
    """``eager`` is valid config (SeedOmni module inference returns before any
    wrap), but ``build_parallelize_model`` has no unwrapped branch."""
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.args = SimpleNamespace(accelerator=AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="eager")))

    with pytest.raises(ValueError, match="eager"):
        runtime._build_parallelized_model()


def test_module_runtime_arguments_inherit_weight_load_knobs():
    args = OmniModuleRuntimeArguments(model_path="/tmp/model")
    assert args.broadcast_model_weights_from_rank0 is True
    assert args.ep_sharded_stream_load is False


def test_validate_omni_accelerator_accepts_defaults():
    _validate_omni_accelerator(AcceleratorConfig())


def test_meta_init_for_fsdp2_is_enforced_at_construction():
    """The rule moved onto ``AcceleratorConfig`` itself, so a module override cannot dodge it.

    ``_validate_omni_accelerator`` no longer repeats it: it could only ever see a
    config that already passed.
    """
    with pytest.raises(AssertionError, match="init_device: meta"):
        AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="fsdp2"), init_device="cpu")


def test_ddp_cpu_init_is_enforced_at_construction():
    with pytest.raises(AssertionError, match="init_device: cpu is not supported"):
        AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="ddp"), init_device="cpu")


def test_ep_sharded_stream_load_with_broadcast_is_enforced_at_construction():
    with pytest.raises(AssertionError, match="ep_sharded_stream_load"):
        OmniModuleRuntimeArguments(
            model_path="/tmp/model",
            ep_sharded_stream_load=True,
            broadcast_model_weights_from_rank0=True,
        )


def test_validate_omni_accelerator_bans_torch_compile():
    acc = AcceleratorConfig(torch_compile=TorchCompileConfig(enable=True))
    with pytest.raises(ValueError, match="torch_compile"):
        _validate_omni_accelerator(acc)


def test_omni_arguments_post_init_validates_the_top_level_default():
    with pytest.raises(ValueError, match="torch_compile"):
        OmniArguments(
            model=OmniModelRuntimeArguments(
                model_path="/tmp/fake_omni",
                accelerator=AcceleratorConfig(torch_compile=TorchCompileConfig(enable=True)),
            ),
            data=OmniDataArguments(train_path=""),
            infer=OmniInferArguments(),
        )


def test_omni_arguments_post_init_refuses_the_moe_router_monitor():
    with pytest.raises(AssertionError, match="moe_load_balance_monitor_interval"):
        OmniArguments(
            model=OmniModelRuntimeArguments(model_path="/tmp/fake_omni"),
            data=OmniDataArguments(train_path=""),
            train=OmniTrainingArguments(moe_load_balance_monitor_interval=10),
            infer=OmniInferArguments(),
        )


def test_build_omni_module_runtime_args_merges_per_module_gradient_checkpointing_override():
    """Global `model.accelerator.gradient_checkpointing` is the base; per-module YAML can override."""
    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/model",
        accelerator=AcceleratorConfig(gradient_checkpointing=GradientCheckpointingConfig(enable=True)),
    )
    modules = build_omni_module_runtime_args(
        global_args,
        "/tmp/model",
        {
            "module_a": {"accelerator": {"gradient_checkpointing": {"enable": False}}},
            "module_b": {"model_path": "module_b"},
        },
    )
    assert modules["module_a"].accelerator.gradient_checkpointing.enable is False
    assert modules["module_b"].accelerator.gradient_checkpointing.enable is True


def test_build_omni_module_runtime_args_merges_per_module_weight_load_override():
    """Global `model.broadcast_*` is the base; a per-module overlay can override it."""
    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/model",
        broadcast_model_weights_from_rank0=True,
    )
    modules = build_omni_module_runtime_args(
        global_args,
        "/tmp/model",
        {
            "module_a": {"broadcast_model_weights_from_rank0": False},
            "module_b": {"model_path": "module_b"},
        },
    )
    assert modules["module_a"].broadcast_model_weights_from_rank0 is False
    assert modules["module_b"].broadcast_model_weights_from_rank0 is True


def test_build_omni_model_runtime_args_accepts_valid_per_module_accelerator_override():
    """A per-module `accelerator.*` override that passes validation resolves cleanly."""
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"gradient_checkpointing": {"enable": False}}}})
    modules = build_omni_model_runtime_args(args).modules
    assert modules[MODULE_B].accelerator.gradient_checkpointing.enable is False
    # Untouched modules keep the top-level default.
    assert modules[MODULE_A].accelerator.gradient_checkpointing.enable is True


@pytest.fixture
def veomni_caplog(caplog):
    """`caplog`, but also attached directly to the ``veomni`` logger.

    The library's root logger (``veomni``) sets ``propagate = False`` (see
    ``veomni/utils/logging.py``) so records never reach the root logger caplog normally
    listens on; attach its handler here so `caplog.text` works for `veomni.*` loggers too.
    """
    logger = logging.getLogger("veomni")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_build_omni_model_runtime_args_for_inference_forces_eager_without_broadcast_warning(veomni_caplog):
    """`for_inference=True` forces `fsdp_mode=eager` for a module that does not pin its own.

    The override replaces `MODULE_B`'s whole `accelerator` block, so the module
    reaches resolution with no `fsdp_config` of its own and inference picks the
    eager default for it; `MODULE_A` pins `ddp` and keeps it.
    `broadcast_model_weights_from_rank0` is forced off alongside eager so the
    default does not inherit a rank0-broadcast load policy that cannot run
    without a wrap (see `build_omni_module_runtime_args`).
    """
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"gradient_checkpointing": {"enable": False}}}})
    with veomni_caplog.at_level("WARNING"):
        modules = build_omni_model_runtime_args(args, for_inference=True).modules
    assert modules[MODULE_B].accelerator.fsdp_config.fsdp_mode == "eager"
    assert modules[MODULE_B].broadcast_model_weights_from_rank0 is False
    assert modules[MODULE_A].accelerator.fsdp_config.fsdp_mode == "ddp"
    assert "broadcast_model_weights_from_rank0" not in veomni_caplog.text


@pytest.mark.parametrize("backend", ["torch", "hyper_parallel"])
@pytest.mark.parametrize("module_backend", [None, "hyper_parallel"])
def test_hyper_training_config_accepts_standard_eager_inference_overlay(module_backend, backend):
    accelerator = AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_backend=backend))
    training_modules = {
        name: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}}} for name in (MODULE_A, MODULE_B)
    }
    if module_backend is not None:
        for module in training_modules.values():
            module["accelerator"]["fsdp_config"]["fsdp_backend"] = module_backend
    args = _fake_args(modules_override=training_modules, accelerator=accelerator)
    training = build_omni_model_runtime_args(args)
    expected_backend = module_backend or backend
    assert all(module.accelerator.fsdp_config.fsdp_backend == expected_backend for module in training.modules.values())
    distributed_inference = build_omni_model_runtime_args(args, for_inference=True)
    assert all(
        module.accelerator.fsdp_config.fsdp_backend == expected_backend
        for module in distributed_inference.modules.values()
    )

    with open(_cfg_dir() / "infer/modules_infer_eager.yaml") as handle:
        eager_overlay = yaml.safe_load(handle)
    for name, overlay in eager_overlay.items():
        args.model.model_config["modules"][name]["accelerator"]["fsdp_config"].update(
            overlay["accelerator"]["fsdp_config"]
        )
    inference = build_omni_model_runtime_args(args, for_inference=True)
    for module in inference.modules.values():
        assert module.accelerator.fsdp_config.fsdp_mode == "eager"
        assert module.accelerator.fsdp_config.fsdp_backend == "torch"
    assert args.model.accelerator.fsdp_config.fsdp_backend == backend


@pytest.mark.parametrize("backend", ["torch", "hyper_parallel"])
def test_omni_setup_does_not_resolve_or_activate_model_configuration(monkeypatch, backend):
    from veomni.trainer.omni import omni_trainer

    args = _fake_args(accelerator=AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_backend=backend)))
    resolve = Mock(side_effect=AssertionError("setup must not resolve model configuration"))
    activate = Mock(side_effect=AssertionError("setup must not activate model adapters"))
    monkeypatch.setattr(omni_trainer, "build_omni_model_runtime_args", resolve)
    monkeypatch.setattr(omni_trainer, "_apply_hyper_parallel_patch", activate)
    accelerator = Mock()
    init_process_group = Mock()
    init_parallel_state = Mock()
    save_args = Mock()
    checkpoint_debug = Mock()
    monkeypatch.setattr(omni_trainer, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(omni_trainer, "get_torch_device", lambda: accelerator)
    monkeypatch.setattr(omni_trainer, "get_dist_comm_backend", lambda: "gloo")
    monkeypatch.setattr(omni_trainer.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(omni_trainer.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(omni_trainer, "init_parallel_state_from_config", init_parallel_state)
    monkeypatch.setattr(omni_trainer, "save_args", save_args)
    monkeypatch.setattr(omni_trainer, "set_checkpoint_debug_enabled", checkpoint_debug)
    for name in ("set_seed", "enable_high_precision_for_bf16", "enable_third_party_logging"):
        monkeypatch.setattr(omni_trainer.helper, name, Mock())

    device = omni_trainer.OmniTrainer._setup(args)

    assert str(device) == f"cpu:{args.train.local_rank}"
    accelerator.set_device.assert_called_once_with(str(device))
    init_process_group.assert_called_once_with(backend="gloo")
    init_parallel_state.assert_called_once_with(args.model.accelerator, name="base")
    omni_trainer.helper.set_seed.assert_called_once_with(args.train.seed, args.train.enable_full_determinism)
    omni_trainer.helper.enable_high_precision_for_bf16.assert_called_once_with()
    if args.train.local_rank == 0:
        omni_trainer.helper.enable_third_party_logging.assert_called_once_with()
    if args.train.global_rank == 0:
        save_args.assert_called_once_with(args, args.train.checkpoint.output_dir)
    checkpoint_debug.assert_called_once_with(args.model.accelerator.gradient_checkpointing.debug)
    resolve.assert_not_called()
    activate.assert_not_called()


@pytest.mark.parametrize("mode", ["eager", "ddp", "fsdp2"])
def test_omni_inference_does_not_import_hyper_for_unused_training_flags(monkeypatch, mode):
    from veomni.trainer.omni import omni_inferencer, omni_trainer

    args = _fake_args(
        accelerator=AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_backend="hyper_parallel")),
        modules_override={
            name: {
                "accelerator": {
                    "fsdp_config": {"fsdp_mode": mode if name == MODULE_A else "eager", "fsdp_backend": "torch"}
                }
            }
            for name in (MODULE_A, MODULE_B)
        },
    )
    args.model.optimizer.type = "muon"
    args.model.optimizer.use_hyper_optimizer = True
    monkeypatch.setitem(sys.modules, "hyper_parallel.integration.veomni", None)
    setup = Mock()
    resolve = Mock(wraps=build_omni_model_runtime_args)
    monkeypatch.setattr(omni_trainer.OmniTrainer, "_setup", setup)
    monkeypatch.setattr(omni_inferencer, "build_omni_model_runtime_args", resolve)
    monkeypatch.setattr(omni_inferencer.helper, "set_seed", Mock())
    monkeypatch.setattr(
        omni_inferencer.OmniInferencer, "_build_model", Mock(side_effect=RuntimeError("stop before model build"))
    )
    with pytest.raises(RuntimeError, match="stop before model build"):
        omni_inferencer.OmniInferencer(args)
    resolve.assert_called_once_with(args, for_inference=True)
    if mode == "eager":
        setup.assert_not_called()
    else:
        setup.assert_called_once_with(args, save_launch_args=False)


@pytest.mark.parametrize(
    "scope,root_mode,module_mode",
    [
        ("module", "fsdp2", "fsdp2"),
        ("model", "fsdp2", "fsdp2"),
        ("model", "fsdp2", "ddp"),
        ("model", "eager", "fsdp2"),
    ],
)
def test_omni_inference_keeps_hyper_for_actual_fsdp_wrapping(monkeypatch, scope, root_mode, module_mode):
    from veomni.trainer.omni import omni_inferencer, omni_trainer

    composed = scope == "model" and root_mode != "eager"
    args = _fake_args(
        accelerator=AcceleratorConfig(
            fsdp_config=FSDPConfig(
                fsdp_backend="hyper_parallel" if composed else "torch", fsdp_scope=scope, fsdp_mode=root_mode
            )
        ),
        modules_override={
            name: {
                "accelerator": {
                    "fsdp_config": {
                        "fsdp_mode": module_mode,
                        "fsdp_backend": "torch" if composed else "hyper_parallel",
                    }
                }
            }
            for name in (MODULE_A, MODULE_B)
        },
    )
    args.model.optimizer.type = "muon"
    args.model.optimizer.use_hyper_optimizer = True
    adapter = ModuleType("hyper_parallel.integration.veomni")
    adapter.apply_hyper_parallel_patch = Mock()
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    setup = Mock()
    monkeypatch.setattr(omni_trainer.OmniTrainer, "_setup", setup)
    monkeypatch.setattr(omni_inferencer.helper, "set_seed", Mock())
    monkeypatch.setattr(
        omni_inferencer.OmniInferencer, "_build_model", Mock(side_effect=RuntimeError("stop before model build"))
    )
    with pytest.raises(RuntimeError, match="stop before model build"):
        omni_inferencer.OmniInferencer(args)
    setup.assert_called_once_with(args, save_launch_args=False)
    assert adapter.apply_hyper_parallel_patch.call_count == 1
    assert adapter.apply_hyper_parallel_patch.call_args.kwargs == {"for_inference": True}


@pytest.mark.parametrize("fsdp,muon", [(True, False), (False, True), (True, True)])
def test_omni_training_does_not_import_hyper_for_overridden_root_defaults(monkeypatch, fsdp, muon):
    from veomni.trainer.base import _apply_hyper_parallel_patch

    args = _fake_args(
        accelerator=AcceleratorConfig(
            fsdp_config=FSDPConfig(fsdp_backend="hyper_parallel" if fsdp else "torch", fsdp_scope="module")
        ),
        modules_override={
            name: {
                "accelerator": {"fsdp_config": {"fsdp_backend": "torch"}},
                "optimizer": {"type": "adamw", "use_hyper_optimizer": False},
            }
            for name in (MODULE_A, MODULE_B)
        },
    )
    args.model.optimizer.type = "muon" if muon else "adamw"
    args.model.optimizer.use_hyper_optimizer = muon
    resolved = build_omni_model_runtime_args(args)
    monkeypatch.setitem(sys.modules, "hyper_parallel.integration.veomni", None)

    _apply_hyper_parallel_patch(resolved)


@pytest.mark.parametrize("fsdp,muon", [(True, False), (False, True), (True, True)])
def test_omni_training_uses_composed_fsdp_and_module_optimizer_selections(monkeypatch, fsdp, muon):
    from veomni.trainer.base import _apply_hyper_parallel_patch

    args = _fake_args(
        accelerator=AcceleratorConfig(
            fsdp_config=FSDPConfig(fsdp_backend="hyper_parallel" if fsdp else "torch", fsdp_scope="model")
        ),
        modules_override={
            name: {
                "accelerator": {"fsdp_config": {"fsdp_backend": "torch"}},
                "optimizer": {"type": "muon" if muon else "adamw", "use_hyper_optimizer": muon},
            }
            for name in (MODULE_A, MODULE_B)
        },
    )
    resolved = build_omni_model_runtime_args(args)
    adapter = ModuleType("hyper_parallel.integration.veomni")
    adapter.apply_hyper_parallel_patch = Mock()
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)

    _apply_hyper_parallel_patch(resolved)

    adapter.apply_hyper_parallel_patch.assert_called_once_with(resolved, for_inference=False)


@pytest.mark.parametrize("fsdp,muon", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("modules_from_file", [False, True])
def test_omni_model_build_reads_module_only_hyper_selection(monkeypatch, tmp_path, fsdp, muon, modules_from_file):
    from veomni.trainer.omni import omni_trainer

    overrides = {
        name: {
            "accelerator": {
                "fsdp_config": {"fsdp_mode": "fsdp2", "fsdp_backend": "hyper_parallel" if fsdp else "torch"}
            }
        }
        for name in (MODULE_A, MODULE_B)
    }
    overrides[MODULE_B]["optimizer"] = {"type": "muon" if muon else "adamw", "use_hyper_optimizer": muon}
    args = _fake_args(modules_override=overrides)
    if modules_from_file:
        modules_path = tmp_path / "modules.yaml"
        modules_path.write_text(yaml.safe_dump(args.model.model_config["modules"]))
        args.model.model_config["modules"] = str(modules_path)
    adapter = ModuleType("hyper_parallel.integration.veomni")
    adapter.apply_hyper_parallel_patch = Mock()
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    resolve = Mock(wraps=build_omni_model_runtime_args)
    monkeypatch.setattr(omni_trainer, "build_omni_model_runtime_args", resolve)
    monkeypatch.setattr(
        omni_trainer, "build_omni_model_runtime", Mock(side_effect=RuntimeError("stop before model build"))
    )
    trainer = omni_trainer.OmniTrainer.__new__(omni_trainer.OmniTrainer)
    trainer.args = args
    with pytest.raises(RuntimeError, match="stop before model build"):
        trainer._build_model_runtime()
    resolve.assert_called_once_with(args)
    assert adapter.apply_hyper_parallel_patch.call_count == int(fsdp or muon)
    if fsdp or muon:
        selected = adapter.apply_hyper_parallel_patch.call_args.args[0]
        assert selected.modules[MODULE_B].accelerator.fsdp_config.fsdp_backend == (
            "hyper_parallel" if fsdp else "torch"
        )
        assert selected.modules[MODULE_B].optimizer.use_hyper_optimizer is muon
    assert args.model.accelerator.fsdp_config.fsdp_backend == "torch"
    assert args.model.optimizer.use_hyper_optimizer is False


def test_build_omni_model_runtime_args_validates_each_module_accelerator():
    """A per-module override that fails `_validate_omni_accelerator` must raise at resolve time.

    The top-level `model.accelerator` default passes validation on its own (no
    `torch_compile.enable`); only the module override sets it, so this only
    fails because `build_omni_model_runtime_args` validates every resolved module's own
    `accelerator`, not just the top-level default.
    """
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"torch_compile": {"enable": True}}}})
    with pytest.raises(ValueError, match="torch_compile"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_rejects_an_eager_module_under_the_composed_wrap():
    """``fsdp_scope='model'`` wraps the composed OmniModel once; an eager module is
    loaded unwrapped, so nothing would shard it."""
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "eager"}}}})
    args.model.accelerator.fsdp_config.fsdp_scope = "model"
    with pytest.raises(ValueError, match=rf"fsdp_mode='eager': \['{MODULE_B}'\]"):
        build_omni_model_runtime_args(args)


def test_deep_update_merges_sp_and_cp_from_base_and_override(monkeypatch):
    """``build_omni_module_runtime_args`` merges module overrides onto ``global_args``
    directly via ``_deep_update``, allowing a module to inherit ``ulysses_size`` from
    base or explicitly switch to ``cp_size`` by resetting ``ulysses_size=1``."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/model",
        accelerator=AcceleratorConfig(ulysses_size=4),
    )
    modules = build_omni_module_runtime_args(
        global_args,
        "/tmp/model",
        {
            "pure_sp": {"model_path": "pure_sp"},
            "pure_cp": {"accelerator": {"cp_size": 4, "ulysses_size": 1}},
        },
    )
    assert (modules["pure_sp"].accelerator.cp_size, modules["pure_sp"].accelerator.ulysses_size) == (1, 4)
    assert (modules["pure_cp"].accelerator.cp_size, modules["pure_cp"].accelerator.ulysses_size) == (4, 1)
    assert modules["pure_sp"].accelerator.dp_size == modules["pure_cp"].accelerator.dp_size == 2


def test_deep_update_unreset_axis_rejected_by_cp_ulysses_guard(monkeypatch):
    """When top-level ``model.accelerator`` sets ``ulysses_size=2`` and a module
    overrides only ``cp_size=2`` without resetting ``ulysses_size=1``, ``_deep_update``
    merges both into ``(cp_size=2, ulysses_size=2)``, which ``AcceleratorConfig``
    rejects because combined CP+Ulysses on a single module is not supported yet."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = _fake_args(
        accelerator=AcceleratorConfig(ulysses_size=2),
        modules_override={
            MODULE_A: {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}}},
            MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}, "cp_size": 2}},
        },
    )
    with pytest.raises(NotImplementedError, match="Context parallelism cannot be combined with Ulysses yet"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_accepts_heterogeneous_sp_and_cp_with_matching_dp_size(monkeypatch):
    """Modules may independently use Ulysses SP or Context Parallelism as long as all
    distributed modules and ``model.accelerator`` resolve to the same ``dp_size``."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = _fake_args(
        accelerator=AcceleratorConfig(ulysses_size=4),
        modules_override={
            MODULE_A: {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}, "ulysses_size": 4, "cp_size": 1}},
            MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}, "cp_size": 4, "ulysses_size": 1}},
        },
    )
    runtime_args = build_omni_model_runtime_args(args)
    assert runtime_args.modules[MODULE_A].accelerator.dp_size == 2
    assert runtime_args.modules[MODULE_B].accelerator.dp_size == 2
    assert args.train.global_batch_size == args.train.micro_batch_size * 2


def test_build_omni_model_runtime_args_rejects_mismatched_module_dp_sizes(monkeypatch):
    """Two distributed modules with different ``dp_size`` must be rejected at argument resolution."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = _fake_args(
        modules_override={
            MODULE_A: {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}, "ulysses_size": 2}},
            MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}, "cp_size": 4, "ulysses_size": 1}},
        }
    )
    with pytest.raises(ValueError, match="All distributed SeedOmni modules must resolve to the same `dp_size`"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_rejects_top_level_dp_size_mismatch(monkeypatch):
    """When top-level ``model.accelerator`` resolves to a ``dp_size`` that disagrees with the modules' ``dp_size``, raise."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = _fake_args(
        modules_override={
            MODULE_A: {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}, "ulysses_size": 4}},
            MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}, "cp_size": 4, "ulysses_size": 1}},
        }
    )
    # Default top-level model.accelerator has ulysses_size=1, cp_size=1 -> dp_size=8 != modules' dp_size=2.
    with pytest.raises(ValueError, match="Top-level `model.accelerator` resolved to dp_size=8"):
        build_omni_model_runtime_args(args)

    # Explicit top-level ulysses_size=2 -> dp_size=4 != modules' dp_size=2.
    args_sp2 = _fake_args(
        accelerator=AcceleratorConfig(ulysses_size=2),
        modules_override={
            MODULE_A: {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}, "ulysses_size": 4}},
            MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}, "cp_size": 4, "ulysses_size": 1}},
        },
    )
    with pytest.raises(ValueError, match="Top-level `model.accelerator` resolved to dp_size=4"):
        build_omni_model_runtime_args(args_sp2)
