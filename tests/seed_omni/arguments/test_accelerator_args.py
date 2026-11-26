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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn
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
    assert acc.skip_sp_gather_between_modules is False
    assert isinstance(acc.gradient_checkpointing, GradientCheckpointingConfig)
    assert isinstance(acc.torch_compile, TorchCompileConfig)
    assert not hasattr(acc, "broadcast_model_weights_from_rank0")
    assert not hasattr(acc, "extra_parallel_stream_load")


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


def test_the_generic_runtime_forwards_extra_parallel_only_to_the_builder(monkeypatch):
    """The no-wrap mode is a builder argument, not something the plan can imply.

    Left unforwarded, an `extra_parallel_only` module would still be FSDP2-wrapped
    -- and a mixed-dtype quantized layer cannot be, so it would fail to load.
    """
    from veomni.distributed import torch_parallelize

    seen = {}

    def fake_build(model, **kwargs):
        seen.update(kwargs)
        return model

    monkeypatch.setattr(torch_parallelize, "build_parallelize_model", fake_build)
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model = nn.Module()
    runtime.train_args = None
    runtime.args = SimpleNamespace(
        accelerator=AcceleratorConfig(
            fsdp_config=FSDPConfig(fsdp_mode="fsdp2", extra_parallel_only=True),
        ),
        basic_modules=[],
        lora_config=None,
        fqn_to_index_mapping=None,
        broadcast_model_weights_from_rank0=False,
        extra_parallel_stream_load=False,
        model_path=None,
        optimizer=SimpleNamespace(type="adamw", muon_expert_zero_comm=False),
    )
    monkeypatch.setattr(VeOmniModelRuntime, "skip_hf_weight_load", False, raising=False)

    runtime._build_parallelized_model()

    assert seen["extra_parallel_only"] is True


@pytest.mark.parametrize(
    "global_accelerator, module_override, expected_mode, expected_flag",
    [
        ({"fsdp_config": FSDPConfig(fsdp_mode="fsdp2", extra_parallel_only=True)}, {}, "eager", True),
        ({"ep_size": 4}, {}, "eager", False),
        (
            {"ep_size": 4},
            {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2", "extra_parallel_only": True}}},
            "fsdp2",
            True,
        ),
        ({}, {}, "eager", False),
        (
            {"fsdp_config": FSDPConfig(fsdp_mode="fsdp2", extra_parallel_only=True)},
            {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}}},
            "ddp",
            True,
        ),
        (
            {"fsdp_config": FSDPConfig(fsdp_mode="fsdp2", extra_parallel_only=True)},
            {"accelerator": {"fsdp_config": {"fsdp_mode": "eager"}}},
            "eager",
            True,
        ),
        (
            {"fsdp_config": FSDPConfig(fsdp_mode="fsdp2", extra_parallel_only=True)},
            {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}}},
            "fsdp2",
            True,
        ),
    ],
    ids=[
        "global_no_wrap_stays_eager",
        "global_ep_stays_eager_without_override",
        "global_ep_explicit_fsdp2",
        "plain_eager",
        "explicit_ddp_ignores_extra_parallel_only",
        "explicit_eager_ignores_extra_parallel_only",
        "explicit_fsdp2",
    ],
)
def test_inference_fsdp_mode_and_extra_parallel_only(
    monkeypatch, global_accelerator, module_override, expected_mode, expected_flag
):
    monkeypatch.setenv("WORLD_SIZE", "8")
    global_args = OmniModuleRuntimeArguments(
        model_path="/tmp/model", accelerator=AcceleratorConfig(**global_accelerator)
    )
    modules = build_omni_module_runtime_args(global_args, "/tmp/model", {"mod": module_override}, for_inference=True)
    mod = modules["mod"]
    assert mod.accelerator.fsdp_config.fsdp_mode == expected_mode
    assert mod.accelerator.fsdp_config.extra_parallel_only is expected_flag


def test_module_runtime_arguments_inherit_weight_load_knobs():
    args = OmniModuleRuntimeArguments(model_path="/tmp/model")
    assert args.broadcast_model_weights_from_rank0 is True
    assert args.extra_parallel_stream_load is False


@pytest.mark.parametrize(
    "override, expected_ep_size",
    [
        ({}, 8),
        ({"accelerator": {"ep_size": 4}}, 4),
        ({"accelerator": {"ep_size": 1}}, 1),
        (
            {
                "accelerator": {
                    "extra_parallel_names": [],
                    "extra_parallel_sizes": [],
                    "extra_parallel_placement_innermost": [],
                }
            },
            8,
        ),
    ],
    ids=["inherited", "shrink_ep4", "disable_ep1", "lists_cleared_keeps_ep"],
)
def test_module_override_ep_size_replaces_or_disables_inherited_ep(monkeypatch, override, expected_ep_size):
    """Per-module ``ep_size`` overrides or disables an inherited global ``ep_size``."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    global_args = OmniModuleRuntimeArguments(model_path="/tmp/model", accelerator=AcceleratorConfig(ep_size=8))
    modules = build_omni_module_runtime_args(global_args, "/tmp/model", {"mod": override})
    resolved = modules["mod"].accelerator
    assert resolved.ep_size == expected_ep_size
    extra_parallel_names, extra_parallel_sizes, _ = resolved.resolved_extra_parallel
    assert dict(zip(extra_parallel_names, extra_parallel_sizes))["ep"] == expected_ep_size


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


def test_validate_omni_accelerator_allows_torch_compile():
    """``torch_compile`` is the switch; what compiles is decided by the config."""
    _validate_omni_accelerator(AcceleratorConfig(torch_compile=TorchCompileConfig(enable=True)))


@pytest.mark.parametrize(
    ("compile_overrides", "accelerator_overrides"),
    [
        ({"backend": "cudagraphs"}, {}),
        ({"mode": "reduce-overhead"}, {}),
        # torch turns on Triton graph capture for this mode too, while
        # ``max-autotune-no-cudagraphs`` deliberately does not.
        ({"mode": "max-autotune"}, {}),
        ({}, {"ulysses_size": 2}),
        ({}, {"cp_size": 2}),
        ({}, {"enable_async": True}),
    ],
)
def test_validate_omni_accelerator_refuses_the_combinations_the_composed_path_cannot_run(
    compile_overrides, accelerator_overrides
):
    """CUDA Graph replay needs a per-step mark the composed trainer does not make,
    and Ulysses slices activations the graph edges reorder.

    ``SimpleNamespace`` rather than an ``AcceleratorConfig``: the dataclass checks
    the mesh in ``__post_init__``, which would reject a 2-way SP size at world
    size 1 before the check under test runs.
    """
    acc = SimpleNamespace(
        torch_compile=TorchCompileConfig(enable=True, **compile_overrides),
        ulysses_size=accelerator_overrides.get("ulysses_size", 1),
        cp_size=accelerator_overrides.get("cp_size", 1),
        enable_async=accelerator_overrides.get("enable_async", False),
    )

    with pytest.raises(ValueError, match="torch_compile"):
        _validate_omni_accelerator(acc)


def test_validate_omni_accelerator_allows_max_autotune_without_cudagraphs():
    """The no-cudagraphs spelling of max-autotune stays available."""
    acc = SimpleNamespace(
        torch_compile=TorchCompileConfig(enable=True, backend="inductor", mode="max-autotune-no-cudagraphs"),
        ulysses_size=1,
        cp_size=1,
        enable_async=False,
    )
    _validate_omni_accelerator(acc)


def test_validate_omni_accelerator_leaves_a_disabled_compile_alone():
    """The SP / CUDA-graph checks only apply when compile is on."""
    acc = SimpleNamespace(
        torch_compile=TorchCompileConfig(enable=False),
        ulysses_size=2,
        cp_size=2,
        enable_async=True,
    )

    _validate_omni_accelerator(acc)


def test_omni_arguments_post_init_accepts_the_top_level_compile_switch():
    """Enabling compile is a valid launcher setting; the per-module flag gates it."""
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


def test_build_omni_model_runtime_args_validates_each_module_accelerator():
    """A per-module override that fails `_validate_omni_accelerator` must raise at resolve time.

    The top-level `model.accelerator` default passes validation on its own; only
    the module override sets an unsupported combination (compile + CUDA Graph
    replay), so this only fails because `build_omni_model_runtime_args` validates
    every resolved module's own `accelerator`, not just the top-level default.
    """
    args = _fake_args(
        modules_override={MODULE_B: {"accelerator": {"torch_compile": {"enable": True, "backend": "cudagraphs"}}}}
    )
    with pytest.raises(ValueError, match="torch_compile"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_allows_a_per_module_compile_switch():
    """Turning compile on for one module is valid; `module_compile_config` decides
    at build time whether that module's resolved wrap can actually run it."""
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"torch_compile": {"enable": True}}}})
    resolved = build_omni_model_runtime_args(args)
    assert resolved.modules[MODULE_B].accelerator.torch_compile.enable is True


def test_build_omni_model_runtime_args_rejects_an_eager_module_under_the_composed_wrap():
    """``fsdp_scope='model'`` wraps the composed OmniModel once; an eager module is
    loaded unwrapped, so nothing would shard it."""
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"fsdp_config": {"fsdp_mode": "eager"}}}})
    args.model.accelerator.fsdp_config.fsdp_scope = "model"
    with pytest.raises(ValueError, match=rf"fsdp_mode='eager': \['{MODULE_B}'\]"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_rejects_a_module_compile_under_the_composed_wrap():
    """``fsdp_scope='model'`` leaves no module able to compile.

    The composed ``OmniModel`` owns the one wrap, so a module cannot wrap itself
    inside it, and the parent cannot compile its children's blocks. Refusing at
    resolve time keeps a per-module overlay from silently compiling nothing.
    """
    args = _fake_args(modules_override={MODULE_B: {"accelerator": {"torch_compile": {"enable": True}}}})
    args.model.accelerator.fsdp_config.fsdp_scope = "model"
    with pytest.raises(ValueError, match=rf"fsdp_scope='model'.*{MODULE_B}"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_rejects_a_launcher_compile_under_the_composed_wrap():
    """The launcher-level switch is refused under ``fsdp_scope='model'`` too."""
    args = _fake_args()
    args.model.accelerator.fsdp_config.fsdp_scope = "model"
    args.model.accelerator.torch_compile.enable = True
    with pytest.raises(ValueError, match=r"fsdp_scope='model'.*model\.accelerator"):
        build_omni_model_runtime_args(args)


def test_build_omni_model_runtime_args_rejects_extra_parallel_only_under_composed_wrap():
    """``fsdp_scope='model'`` rejects both top-level and per-module ``extra_parallel_only=True``."""
    # Root extra_parallel_only=True must be rejected even when every child overrides it to False,
    # because child ParallelStates under fsdp_scope='model' are built from the top-level accelerator.
    args = _fake_args(
        modules_override={
            MODULE_A: {"accelerator": {"fsdp_config": {"extra_parallel_only": False}}},
            MODULE_B: {"accelerator": {"fsdp_config": {"extra_parallel_only": False}}},
        }
    )
    args.model.accelerator.fsdp_config.fsdp_scope = "model"
    args.model.accelerator.fsdp_config.extra_parallel_only = True
    with pytest.raises(ValueError, match="top-level fsdp_config.extra_parallel_only"):
        build_omni_model_runtime_args(args)

    # Per-module extra_parallel_only=True under root fsdp_scope='model' is also rejected.
    child_args = _fake_args(
        modules_override={MODULE_B: {"accelerator": {"fsdp_config": {"extra_parallel_only": True}}}}
    )
    child_args.model.accelerator.fsdp_config.fsdp_scope = "model"
    with pytest.raises(ValueError, match=rf"fsdp_config.extra_parallel_only: \['{MODULE_B}'\]"):
        build_omni_model_runtime_args(child_args)


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


def test_skip_sp_gather_between_modules_is_global_and_propagates_to_modules(monkeypatch):
    """``skip_sp_gather_between_modules`` is a run-wide switch propagated to every module.

    It describes the whole graph -- every ``post_forward`` either gathers or hands
    its SP shard on -- so a per-module YAML entry cannot desynchronise the two
    sides of a handoff; the top-level value wins.
    """
    from veomni.distributed import parallel_state
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config

    class _DummyMesh:
        def __getitem__(self, _key):
            return self

        def _flatten(self, **_kwargs):
            return self

    monkeypatch.setattr(parallel_state, "init_device_mesh", lambda *a, **kw: _DummyMesh())
    args = _fake_args(
        accelerator=AcceleratorConfig(skip_sp_gather_between_modules=True),
        modules_override={
            MODULE_A: {"accelerator": {"skip_sp_gather_between_modules": False}},
        },
    )
    resolved = build_omni_model_runtime_args(args)
    assert resolved.accelerator.skip_sp_gather_between_modules is True
    for mod_args in resolved.modules.values():
        assert mod_args.accelerator.skip_sp_gather_between_modules is True

    clear_parallel_state()
    try:
        ps = init_parallel_state_from_config(resolved.accelerator, name="test_skip_sp_gather")
        assert ps.skip_sp_gather_between_modules is True
    finally:
        clear_parallel_state()
