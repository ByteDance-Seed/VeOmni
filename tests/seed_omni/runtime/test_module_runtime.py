# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``ModuleRuntime`` is a ``VeOmniModelRuntime``.

What is tested here is the seam, not the build: that the omni runtime reuses the
base build sequence, and that the handful of places a *module* legitimately
differs from a standalone model are the places that override it.
"""

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch.nn as nn
from torch.utils.checkpoint import noop_context_fn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import GradientCheckpointingConfig
from veomni.distributed import parallel_state
from veomni.distributed.parallel_state import ParallelState, use_parallel_state
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.models.seed_omni.accelerated.omni_module import omni_module_runtime
from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from veomni.models.seed_omni.mixins.offline_encoding_mixin import OfflineEncodingMixin
from veomni.utils import recompute_utils


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _fsdp(scope: str, mode: str = "fsdp2") -> SimpleNamespace:
    return SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_scope=scope, fsdp_mode=mode))


def _unbuilt(model: nn.Module | None = None, **args_fields) -> ModuleRuntime:
    """A ModuleRuntime with its fields set but no build run."""
    args_fields.setdefault("accelerator", _fsdp("module"))
    args_fields.setdefault("basic_modules", [])
    runtime = ModuleRuntime.__new__(ModuleRuntime)
    runtime._global_accelerator = _fsdp("module")
    runtime.model = model
    runtime.model_name = "vision_encoder"
    runtime.args = SimpleNamespace(model_path="/tmp/hf-model", lora_config=None, **args_fields)
    runtime.train_args = None
    return runtime


def test_module_runtime_is_a_model_runtime():
    assert issubclass(ModuleRuntime, VeOmniModelRuntime)


def test_omni_runtime_arguments_are_model_arguments():
    from veomni.arguments import ModelArguments
    from veomni.arguments.omni_arguments_types import OmniModelRuntimeArguments, OmniModuleRuntimeArguments

    assert issubclass(OmniModuleRuntimeArguments, ModelArguments)
    assert issubclass(OmniModelRuntimeArguments, ModelArguments)


def test_module_checkpoint_manager_is_a_model_checkpoint_manager():
    from veomni.models.checkpoint import ModelCheckpointManager
    from veomni.models.seed_omni.utils.checkpoint import OmniModuleCheckpointManager

    assert issubclass(OmniModuleCheckpointManager, ModelCheckpointManager)


def test_module_name_is_the_base_model_name():
    """One identity: the ParallelState registry key, the checkpoint subdir, the graph node."""
    runtime = _unbuilt()
    assert runtime.module_name == runtime.model_name == "vision_encoder"


def test_module_name_is_read_only_so_the_two_names_cannot_diverge():
    with pytest.raises(AttributeError):
        _unbuilt().module_name = "something_else"


@pytest.mark.parametrize(
    "member",
    [
        "setup",  # a module's mesh is built from its accelerator like any model's
        "_setup_lora",
        "parallel_state",
    ],
)
def test_shared_build_steps_are_not_reimplemented(member):
    assert member not in vars(ModuleRuntime), f"{member} should be inherited, not duplicated"


@pytest.mark.parametrize(
    "method",
    [
        "_build_model",  # config lives beside the module's weights, not at the omni root
        "_build_model_assets",  # bound onto the model, because the graph calls the module
        "_freeze_model_module",  # the report has to name which module it describes
        "_build_parallelized_model",  # a custom runtime may own the wrap
        "_build_optimizer",  # frozen modules get none; scoped to the module's mesh
        "_build_lr_scheduler",
        "build_checkpoint",  # per-module manager, absent when frozen
        "clip_grad_norm",  # returns this module's norm; the orchestrator combines
        "skip_hf_weight_load",
        "on_lora_matched_nothing",  # a module the config skipped is not an error
        "save_model_assets",  # the composed root's sidecars are not a module's job
        "__call__",  # a direct forward still has to enter the module's mesh
    ],
)
def test_module_specific_steps_are_overridden(method):
    assert method in vars(ModuleRuntime), f"{method} must state how a module differs"


def test_build_model_uses_the_config_the_omni_config_loaded(monkeypatch):
    """A module never reads its own ``config.json``: the composed OmniConfig loaded
    it and hands it in. ``model_path`` is only where the weights live."""
    captured = {}

    def fake_build_foundation_model(**kwargs):
        captured.update(kwargs)
        model = nn.Linear(2, 2)
        model.config = SimpleNamespace()
        return model

    monkeypatch.setattr("veomni.models.build_foundation_model", fake_build_foundation_model)

    runtime = _unbuilt(
        config_path="/tmp/omni-checkpoint-root",
        model_config=None,
        ops_implementation=None,
        accelerator=SimpleNamespace(
            init_device="meta",
            fsdp_config=SimpleNamespace(
                fsdp_mode="fsdp2", fsdp_scope="module", mixed_precision=SimpleNamespace(enable=False)
            ),
        ),
    )
    runtime.module_config = SimpleNamespace(model_type="fake", ops_config=None)
    runtime._build_model()

    assert captured["config_path"] is runtime.module_config
    assert captured["weights_path"] == "/tmp/hf-model"


class _CacheableModule(OfflineEncodingMixin, nn.Linear):
    pass


@pytest.mark.parametrize(
    ("training_task", "model_cls", "on_meta"),
    [
        ("offline_training", _CacheableModule, True),
        ("offline_training", nn.Linear, False),
        ("offline_embedding", _CacheableModule, False),
        ("online_training", _CacheableModule, False),
        (None, _CacheableModule, False),
    ],
)
def test_only_a_cacheable_module_is_built_on_meta_for_offline_training(monkeypatch, training_task, model_cls, on_meta):
    """``offline_training`` reads every module that mixes in ``OfflineEncodingMixin`` from the cache,
    whichever of its methods the graph names; no other module and no other task does.

    ``None`` stands for an inference build, which has no train args.
    """
    captured = {}

    def fake_build_foundation_model(**kwargs):
        captured.update(kwargs)
        model = nn.Linear(2, 2)
        model.config = SimpleNamespace()
        return model

    monkeypatch.setattr("veomni.models.build_foundation_model", fake_build_foundation_model)
    monkeypatch.setattr("veomni.models.loader.get_model_class", lambda config: model_cls)
    runtime = _offline_cache_runtime(training_task)

    assert runtime.reads_offline_cache is on_meta
    runtime._build_model()

    assert captured["init_device"] == ("meta" if on_meta else "cpu")


@pytest.mark.parametrize(("accelerated_cls", "on_meta"), [(_CacheableModule, True), (None, False)])
def test_the_class_the_omni_registries_build_decides_the_meta_build(monkeypatch, accelerated_cls, on_meta):
    """The real lookup prefers the accelerated class, so a marker there (as on BAGEL's VAE) is what counts."""
    from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_MODEL_REGISTRY

    monkeypatch.delenv("MODELING_BACKEND", raising=False)
    model_type = "offline_cache_marker_test"
    OMNI_MODEL_REGISTRY[model_type] = lambda: nn.Linear
    if accelerated_cls is not None:
        OMNI_ACCELERATED_MODEL_REGISTRY[model_type] = lambda: accelerated_cls
    try:
        runtime = _offline_cache_runtime("offline_training")
        runtime.module_config = SimpleNamespace(model_type=model_type, architectures=None)
        assert runtime.reads_offline_cache is on_meta
    finally:
        del OMNI_MODEL_REGISTRY[model_type]
        if accelerated_cls is not None:
            del OMNI_ACCELERATED_MODEL_REGISTRY[model_type]


def test_a_cache_reading_module_is_frozen_and_never_wrapped_trained_or_saved(monkeypatch):
    calls = []
    for step in (
        "setup",
        "_freeze_model_module",
        "_build_parallelized_model",
        "_build_optimizer",
        "build_checkpoint",
    ):
        monkeypatch.setattr(ModuleRuntime, step, lambda self, *a, _step=step, **k: calls.append(_step))
    monkeypatch.setattr(ModuleRuntime, "_build_model_assets", lambda self: None)
    monkeypatch.setattr(ModuleRuntime, "_scoped", lambda self: nullcontext())

    def fake_build_model(self):
        self.model = nn.Linear(2, 2)

    monkeypatch.setattr(ModuleRuntime, "_build_model", fake_build_model)
    monkeypatch.setattr("veomni.models.loader.get_model_class", lambda config: _CacheableModule)
    train = SimpleNamespace(training_task="offline_training", checkpoint=SimpleNamespace(load_path=None))

    runtime = ModuleRuntime(
        SimpleNamespace(accelerator=_fsdp("module")),
        "vae",
        module_config=SimpleNamespace(),
        global_accelerator=_fsdp("module"),
        train_args=train,
    )

    assert calls == ["setup"]
    assert not any(p.requires_grad for p in runtime.model.parameters())


def test_an_offline_embedding_run_loads_and_wraps_every_module_but_freezes_it(monkeypatch):
    """It trains nothing, so the frozen module gets no optimizer and no checkpoint manager."""
    calls = []
    for step in ("setup", "_freeze_model_module", "_build_parallelized_model"):
        monkeypatch.setattr(ModuleRuntime, step, lambda self, *a, _step=step, **k: calls.append(_step))
    monkeypatch.setattr(ModuleRuntime, "_build_model_assets", lambda self: None)
    monkeypatch.setattr(ModuleRuntime, "_scoped", lambda self: nullcontext())
    monkeypatch.setattr(ModuleRuntime, "_build_model", lambda self: setattr(self, "model", nn.Linear(2, 2)))
    train = SimpleNamespace(training_task="offline_embedding", checkpoint=SimpleNamespace(load_path=None))

    runtime = ModuleRuntime(
        SimpleNamespace(accelerator=_fsdp("module")),
        "llm",
        module_config=SimpleNamespace(),
        global_accelerator=_fsdp("module"),
        train_args=train,
    )

    assert calls == [
        "setup",
        "_freeze_model_module",
        "_build_parallelized_model",
    ]
    assert not runtime.has_trainable_parameters
    assert runtime.optimizer is None
    assert getattr(runtime, "checkpoint", None) is None


def _offline_cache_runtime(training_task):
    runtime = _unbuilt(
        model_config=None,
        ops_implementation=None,
        accelerator=SimpleNamespace(
            init_device="cpu",
            fsdp_config=SimpleNamespace(
                fsdp_mode="fsdp2", fsdp_scope="module", mixed_precision=SimpleNamespace(enable=False)
            ),
        ),
    )
    runtime.module_config = SimpleNamespace(model_type="fake", ops_config=None)
    if training_task is not None:
        runtime.train_args = SimpleNamespace(training_task=training_task)
    return runtime


# The base declares these as class attributes defaulting to ``None``, so a test
# asserting ``is None`` after the call would pass even if the builder did nothing.
# Seeding a sentinel makes the assertion carry the signal.
_UNSET = object()


def test_a_frozen_module_builds_no_optimizer():
    runtime = _unbuilt(nn.Linear(2, 2).requires_grad_(False))
    runtime.optimizer = _UNSET

    runtime._build_optimizer()

    assert runtime.optimizer is _UNSET, "the builder must return before touching the optimizer"


def test_a_frozen_module_builds_no_lr_scheduler():
    runtime = _unbuilt(nn.Linear(2, 2).requires_grad_(False))
    runtime.lr_scheduler = _UNSET

    runtime._build_lr_scheduler(total_steps=100)

    assert runtime.lr_scheduler is _UNSET


def test_a_frozen_module_gets_no_checkpoint_manager():
    runtime = _unbuilt(nn.Linear(2, 2).requires_grad_(False))
    runtime.checkpoint = _UNSET

    runtime.build_checkpoint()

    assert runtime.checkpoint is None
    assert runtime.has_trainable_parameters is False


def test_a_trainable_module_does_get_a_checkpoint_manager(monkeypatch):
    """The frozen carve-out must be a carve-out, not the only path."""
    built = []
    monkeypatch.setattr(
        "veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.OmniModuleCheckpointManager",
        lambda runtime: built.append(runtime) or "manager",
    )

    runtime = _unbuilt(nn.Linear(2, 2))
    runtime.build_checkpoint()

    assert runtime.checkpoint == "manager"
    assert built == [runtime]


def test_the_real_checkpoint_manager_reads_the_modules_train_args():
    """The base manager reads ``runtime.train_args``; a module that stored its
    training args under another name would resolve it on the wrapped model."""
    from veomni.models.seed_omni.utils.checkpoint import OmniModuleCheckpointManager

    runtime = _unbuilt(nn.Linear(2, 2), accelerator=_fsdp("module"))
    checkpoint = SimpleNamespace(manager="dcp", load_path=None)
    runtime.train_args = SimpleNamespace(checkpoint=checkpoint)

    manager = OmniModuleCheckpointManager(runtime)

    assert manager.config is checkpoint
    assert manager.module_name == "vision_encoder"


def test_the_module_checkpoint_manager_nests_every_artifact_under_the_module(monkeypatch):
    """The base stops at the step directory; only the module manager adds a level."""
    from veomni.models.seed_omni.utils.checkpoint import OmniModuleCheckpointManager
    from veomni.trainer.callbacks import TrainerState

    runtime = _unbuilt(nn.Linear(2, 2), accelerator=_fsdp("module"))
    checkpoint = SimpleNamespace(manager="dcp", load_path=None, save_path="/run", model_assets_dir="/out/model_assets")
    runtime.train_args = SimpleNamespace(checkpoint=checkpoint)
    runtime.args.lora_config = None
    monkeypatch.setattr(ModuleRuntime, "parallel_state", property(lambda self: "vision_ps"))

    manager = OmniModuleCheckpointManager(runtime)
    state = TrainerState(global_step=3)

    assert manager.save_dir(state) == "/run/global_step_3/model/vision_encoder"
    assert manager.weights_dir(state) == "/run/global_step_3/model/vision_encoder/ckpt"
    assert manager.hf_export_dir(state) == "/run/global_step_3/hf_ckpt/vision_encoder"
    assert manager.lora_export_dir(state) == "/run/global_step_3/lora_ckpt/vision_encoder"
    assert manager.assets_dir() == "/out/model_assets/vision_encoder"
    assert manager._checkpointer_kwargs() == {
        "trainable_only": False,
        "parallel_state": "vision_ps",
        "module": "vision_encoder",
    }


def test_the_constructor_stores_training_args_where_the_base_reads_them(monkeypatch):
    for step in (
        "setup",
        "_build_model",
        "_build_model_assets",
        "_freeze_model_module",
        "_build_parallelized_model",
        "_build_optimizer",
        "build_checkpoint",
    ):
        monkeypatch.setattr(ModuleRuntime, step, lambda self, *a, **k: None)
    monkeypatch.setattr(ModuleRuntime, "_scoped", lambda self: nullcontext())
    train = SimpleNamespace(training_task="online_training", checkpoint=SimpleNamespace(load_path=None))
    args = SimpleNamespace(accelerator=_fsdp("module"))

    runtime = ModuleRuntime(
        args, "vision_encoder", module_config=SimpleNamespace(), global_accelerator=_fsdp("module"), train_args=train
    )

    assert vars(runtime)["train_args"] is train


@pytest.mark.parametrize(
    ("top_level", "overlay", "wraps_omni_model"), [("model", "module", True), ("module", "model", False)]
)
def test_only_the_top_level_scope_decides_wrap_omni_model(monkeypatch, top_level, overlay, wraps_omni_model):
    """The composed wrap reads the top-level scope, so a module reading its own
    overlay would be wrapped twice, or left on meta with nothing to wrap it."""
    for step in (
        "setup",
        "_build_model",
        "_build_model_assets",
        "_freeze_model_module",
        "_build_parallelized_model",
        "_build_optimizer",
        "build_checkpoint",
    ):
        monkeypatch.setattr(ModuleRuntime, step, lambda self, *a, **k: None)
    monkeypatch.setattr(ModuleRuntime, "_scoped", lambda self: nullcontext())
    args = SimpleNamespace(accelerator=_fsdp(overlay))

    runtime = ModuleRuntime(
        args, "vision_encoder", module_config=SimpleNamespace(), global_accelerator=_fsdp(top_level)
    )

    assert runtime.wrap_omni_model is wraps_omni_model


def test_a_module_under_the_omni_model_wrap_applies_async_activation_offload_first(monkeypatch):
    """The patching must precede ``fully_shard``, which under ``fsdp_scope='model'`` the
    composed OmniModel runs later; the module's own parallelize step is the last chance."""
    applied = []
    monkeypatch.setattr(ModuleRuntime, "_apply_async_activation_offload", lambda self: applied.append(self))
    runtime = _unbuilt(nn.Linear(2, 2))
    runtime._global_accelerator = _fsdp("model")

    runtime._build_parallelized_model()

    assert applied == [runtime]


class _CheckpointableLinear(nn.Linear):
    """An HF-style model: it owns the ``gradient_checkpointing_enable`` entry point."""

    def __init__(self, log=None):
        super().__init__(2, 2)
        self.log = [] if log is None else log

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.log.append(("enable", gradient_checkpointing_kwargs))


class _RecordingParallelState:
    """Stands in for ``use_parallel_state(name)``, recording where it was entered."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __enter__(self):
        self.log.append(self.name)
        return self

    def __exit__(self, *exc_info):
        return False


def _recompute_accelerator(**gc_fields):
    return SimpleNamespace(
        gradient_checkpointing=GradientCheckpointingConfig(**gc_fields),
        offload_config=SimpleNamespace(enable_activation=False, enable_async_activation=False),
        torch_compile=SimpleNamespace(enable=False),
    )


def test_the_recompute_apply_enables_first_then_applies_the_policy(monkeypatch):
    """The module's ParallelState is folded into the policy, never armed as a second enable.

    Enabling again *after* the apply rewrites every block's entry point, which is
    how the composed path used to lose both the layer filter and SAC.
    """
    calls = []
    model = _CheckpointableLinear(calls)
    monkeypatch.setattr(
        recompute_utils,
        "apply_recompute_policy",
        lambda wrapped, policy, **kwargs: calls.append(("apply", wrapped, policy)),
    )
    accelerator = _recompute_accelerator(enable=True, recompute_layers=1, selective_recompute_layers=1)

    _unbuilt(model, accelerator=accelerator)._apply_recompute_policy()

    assert [call[0] for call in calls] == ["enable", "apply"]
    enable_kwargs = calls[0][1]
    assert enable_kwargs["use_reentrant"] is False and enable_kwargs["early_stop"] is True

    _, wrapped, policy = calls[1]
    assert wrapped is model
    assert (policy.recompute_layers, policy.selective_recompute_layers) == (1, 1)
    assert isinstance(policy.recompute_context, partial)
    assert policy.recompute_context.func is use_parallel_state
    assert policy.recompute_context.args == ("vision_encoder",)


def test_the_enable_scopes_recompute_to_the_modules_parallel_state(monkeypatch):
    """Blocks the policy does not bind keep HF's function, so the enable must carry
    the module's ParallelState itself — the default config binds nothing."""
    entered = []
    model = _CheckpointableLinear()
    monkeypatch.setattr(omni_module_runtime, "use_parallel_state", lambda name: _RecordingParallelState(name, entered))
    accelerator = _recompute_accelerator(enable=True)  # the defaults: nothing is bound selectively

    _unbuilt(model, accelerator=accelerator)._apply_recompute_policy()

    context_fn = model.log[0][1]["context_fn"]
    assert context_fn is not noop_context_fn
    forward_ctx, recompute_ctx = context_fn()
    with forward_ctx:
        assert entered == []
    with recompute_ctx:
        assert entered == ["vision_encoder"]


def test_the_enable_keeps_torchs_default_without_a_parallel_state(monkeypatch):
    """A module with no state of its own has nothing to fold in."""
    model = _CheckpointableLinear()
    monkeypatch.setattr(ModuleRuntime, "_recompute_context", lambda self: None)
    accelerator = _recompute_accelerator(enable=True)

    _unbuilt(model, accelerator=accelerator)._apply_recompute_policy()

    assert model.log[0][1]["context_fn"] is noop_context_fn


def test_the_recompute_apply_ranks_the_stacks_by_the_modules_basic_modules(monkeypatch):
    """Ranking the block stacks is what ``basic_modules`` is for, and the composed scope
    has to rank them like the ``fsdp_scope='module'`` wrap does, or one configuration
    would bind different stacks in the two scopes."""
    seen = []
    model = _CheckpointableLinear()
    monkeypatch.setattr(
        recompute_utils,
        "apply_recompute_policy",
        lambda wrapped, policy, **kwargs: seen.append(kwargs),
    )
    accelerator = _recompute_accelerator(enable=True)

    _unbuilt(model, accelerator=accelerator, basic_modules=["BlockA"])._apply_recompute_policy()

    assert seen == [{"basic_modules": ["BlockA"]}]


def test_reentrant_checkpointing_is_rejected_for_a_scoped_module(monkeypatch):
    """Reentrant checkpointing takes no ``context_fn``, so a module that must recompute
    inside its ParallelState cannot use it: the run would recompute with the wrong state
    current. Rejected, not silently scoped out."""
    model = _CheckpointableLinear()
    accelerator = _recompute_accelerator(enable=True, enable_reentrant=True)

    with pytest.raises(ValueError, match="enable_reentrant"):
        _unbuilt(model, accelerator=accelerator)._apply_recompute_policy()

    assert model.log == []


def test_reentrant_checkpointing_is_only_rejected_while_checkpointing_is_on():
    """``enable_reentrant`` says nothing with checkpointing off: no reentrant
    checkpoint call happens, so the policy is built instead of failing the run."""
    accelerator = _recompute_accelerator(enable=False, enable_reentrant=True)

    policy = _unbuilt(accelerator=accelerator)._build_recompute_policy()

    assert policy.use_reentrant is True


def test_the_recompute_apply_is_a_no_op_when_checkpointing_is_off(monkeypatch):
    calls = []
    model = _CheckpointableLinear(calls)
    monkeypatch.setattr(recompute_utils, "apply_recompute_policy", lambda *args, **kwargs: calls.append(("apply",)))
    accelerator = SimpleNamespace(gradient_checkpointing=GradientCheckpointingConfig(enable=False))

    _unbuilt(model, accelerator=accelerator)._apply_recompute_policy()

    assert calls == []


def test_distributed_inference_wraps_lora_before_loading_weights(monkeypatch):
    """With ``lora_config`` set the loader maps base keys onto ``base_model.model.*``;
    an unwrapped model has no such names, so its base weights would never load."""
    calls = []
    for step in ("setup", "_build_model_assets", "_setup_lora", "_build_parallelized_model"):
        monkeypatch.setattr(ModuleRuntime, step, lambda self, *a, _step=step, **k: calls.append(_step))
    monkeypatch.setattr(ModuleRuntime, "_build_model", lambda self: setattr(self, "model", nn.Linear(2, 2)))
    monkeypatch.setattr(ModuleRuntime, "_scoped", lambda self: nullcontext())
    accelerator = _fsdp("module")
    accelerator.fsdp_config.mixed_precision = SimpleNamespace(enable=True)

    ModuleRuntime(
        SimpleNamespace(accelerator=accelerator),
        "llm",
        module_config=SimpleNamespace(),
        global_accelerator=_fsdp("module"),
        for_inference=True,
    )

    assert calls.index("_setup_lora") < calls.index("_build_parallelized_model")


def test_a_module_the_lora_config_missed_stays_frozen_instead_of_failing():
    """A composed ``lora_config`` reaches every module, so "no targets here" is
    how it says which model to adapt — only the composer can call it an error."""
    runtime = _unbuilt(nn.Linear(2, 2))

    runtime.on_lora_matched_nothing()  # must not raise


def test_a_module_does_not_write_the_jobs_model_assets():
    with pytest.raises(NotImplementedError, match="OmniModelRuntime.save_model_assets"):
        _unbuilt(nn.Linear(2, 2)).save_model_assets()


def test_model_assets_are_read_off_the_live_model():
    """A module's sidecars are bound onto the model, so an asset bound after the
    build still reaches the HF export."""
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace()
    runtime = _unbuilt(model)
    model._tokenizer = SimpleNamespace()

    assert runtime.model_assets == [model.config, model._tokenizer]


class _RecordAmbientState(nn.Module):
    """Records whichever ParallelState was current when its forward ran."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = "never called"

    def forward(self, value):
        self.seen = parallel_state._PARALLEL_STATE
        return value


@pytest.fixture
def registered_state():
    """Register a distinguishable ParallelState under the runtime's module name."""
    state = ParallelState()
    parallel_state._PARALLEL_STATE_REGISTRY["vision_encoder"] = state
    try:
        yield state
    finally:
        parallel_state._PARALLEL_STATE_REGISTRY.pop("vision_encoder", None)
        parallel_state.set_parallel_state(None)


def test_a_direct_forward_runs_inside_the_modules_own_parallel_state(registered_state):
    """Attention resolves its all-to-all group from the *current* state, so an
    unscoped forward under Ulysses SP would communicate over the wrong ranks —
    silent corruption, not an error."""
    model = _RecordAmbientState()
    runtime = _unbuilt(model)
    orchestrator_state = ParallelState()
    parallel_state.set_parallel_state(orchestrator_state)

    assert runtime(7) == 7
    assert model.seen is registered_state
    assert parallel_state._PARALLEL_STATE is orchestrator_state, "the ambient state must be restored"


def test_an_eager_inference_module_forwards_without_a_parallel_state(monkeypatch):
    """An eager-inference module never runs ``setup``, so it has no state to
    enter; its scope is a no-op rather than a registry lookup that would fail."""
    model = _RecordAmbientState()
    monkeypatch.setattr(ModuleRuntime, "_init_eager_inference", lambda self: setattr(self, "model", model))
    args = SimpleNamespace(accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="eager")))

    runtime = ModuleRuntime(
        args, "vision_encoder", module_config=SimpleNamespace(), global_accelerator=_fsdp("module"), for_inference=True
    )

    assert "vision_encoder" not in parallel_state._PARALLEL_STATE_REGISTRY
    assert runtime(7) == 7
    assert model.seen != "never called"


def test_a_frozen_module_has_no_async_save_to_drain():
    runtime = ModuleRuntime.__new__(ModuleRuntime)
    runtime.checkpoint = None

    runtime.wait_for_pending_save()


def test_compile_is_off_by_default_for_a_module():
    """A module that has not been measured never compiles.

    ``supports_torch_compile`` is the per-module gate; the class default has to be
    ``False`` or every module would inherit the run switch.
    """
    from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule

    assert PretrainedOmniModule.supports_torch_compile is False


@pytest.mark.parametrize(
    ("run_enable", "module_support", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_module_compile_config_needs_both_switches(run_enable, module_support, expected):
    """The run switch and the module's opt-in are ANDed.

    The run switch alone must not compile a module that never opted in, and a
    module's opt-in must not compile itself in a run that left compile off.
    """
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import module_compile_config

    module = SimpleNamespace(supports_torch_compile=module_support)
    accelerator = SimpleNamespace(
        torch_compile=SimpleNamespace(enable=run_enable, backend="inductor", mode=None, fullgraph=True, dynamic=False),
        fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"),
    )

    assert module_compile_config(accelerator, module).enable is expected


def test_module_compile_config_carries_the_run_options_through():
    """Gating changes ``enable`` only; the chosen backend and shape flags survive."""
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import module_compile_config

    accelerator = SimpleNamespace(
        torch_compile=SimpleNamespace(enable=True, backend="inductor", mode=None, fullgraph=False, dynamic=True),
        fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"),
    )
    module = SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=True)
    config = module_compile_config(accelerator, module)

    assert (config.backend, config.mode, config.fullgraph, config.dynamic) == ("inductor", None, False, True)


def test_a_module_without_the_flag_is_treated_as_opted_out():
    """``getattr`` default, so a module class predating the flag stays eager."""
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import module_compile_config

    accelerator = SimpleNamespace(
        torch_compile=SimpleNamespace(enable=True, backend="inductor", mode=None, fullgraph=True, dynamic=False),
        fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"),
    )

    assert module_compile_config(accelerator, SimpleNamespace()).enable is False


def test_compile_dynamic_is_off_by_default_for_a_module():
    """A module opts into ``dynamic`` separately from compile itself.

    ``dynamic`` pays Inductor's polymorphic-kernel tax, which on a kernel-heavy
    codec exceeded the compile win entirely (``janus_vqvae``: 30% slower than
    static), so it cannot be inherited from the run switch.
    """
    from veomni.models.seed_omni.modules.module_modeling_base import PretrainedOmniModule

    assert PretrainedOmniModule.supports_torch_compile_dynamic is False


@pytest.mark.parametrize(
    ("run_dynamic", "module_dynamic", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_module_compile_config_intersects_dynamic(run_dynamic, module_dynamic, expected):
    """The run-level ``dynamic`` only reaches modules that declared they can use it."""
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import module_compile_config

    module = SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=module_dynamic)
    accelerator = SimpleNamespace(
        torch_compile=SimpleNamespace(enable=True, backend="inductor", mode=None, fullgraph=True, dynamic=run_dynamic),
        fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"),
    )

    assert module_compile_config(accelerator, module).dynamic is expected


def test_a_module_without_the_dynamic_flag_gets_dynamic_off():
    """``getattr`` default, so a module class predating the flag never turns it on."""
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import module_compile_config

    accelerator = SimpleNamespace(
        torch_compile=SimpleNamespace(enable=True, backend="inductor", mode=None, fullgraph=True, dynamic=True),
        fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"),
    )

    assert module_compile_config(accelerator, SimpleNamespace(supports_torch_compile=True)).dynamic is False


def _fake_runtime(model, *, fsdp_mode="fsdp2", enable=True):
    """A ``ModuleRuntime`` stub carrying just what the scope helpers read.

    ``enable`` is the module's own resolved ``accelerator.torch_compile.enable``,
    which a per-module YAML overlay can set.
    """
    return SimpleNamespace(
        model=model,
        args=SimpleNamespace(
            accelerator=SimpleNamespace(
                torch_compile=SimpleNamespace(enable=enable),
                fsdp_config=SimpleNamespace(fsdp_mode=fsdp_mode),
            )
        ),
    )


@pytest.mark.parametrize(
    ("support", "dynamic", "fsdp_mode", "compilable", "dynamic_scope"),
    [
        (True, True, "fsdp2", True, True),
        (True, False, "fsdp2", True, False),
        (False, True, "fsdp2", False, False),
        # ddp cannot compile at all: `compile_decoder_blocks` only runs under fsdp2
        (True, True, "ddp", False, False),
    ],
)
def test_compilable_scope_intersects_the_three_gates(support, dynamic, fsdp_mode, compilable, dynamic_scope):
    """Compiling needs the module's opt-in *and* fsdp2; ``dynamic`` is a third, separate opt-in.

    The ``ddp`` row is the shipped ``janus_siglip`` case: it declares
    ``supports_torch_compile`` but runs under DDP, so it has to stay eager
    rather than fail its wrap.
    """
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        _compilable_scope,
        _dynamic_scope,
    )

    model = SimpleNamespace(supports_torch_compile=support, supports_torch_compile_dynamic=dynamic)
    runtimes = {"m": _fake_runtime(model, fsdp_mode=fsdp_mode)}

    assert bool(_compilable_scope(runtimes)) is compilable
    assert bool(_dynamic_scope(runtimes)) is dynamic_scope


def test_dynamic_scope_is_a_subset_of_the_compilable_scope():
    """A module that does not compile is never offered the ``dynamic`` path."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        _compilable_scope,
        _dynamic_scope,
    )

    runtimes = {
        "llm": _fake_runtime(SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=True)),
        "vqvae": _fake_runtime(SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=False)),
        "tower": _fake_runtime(SimpleNamespace(supports_torch_compile=False, supports_torch_compile_dynamic=True)),
    }

    assert set(_dynamic_scope(runtimes)) <= set(_compilable_scope(runtimes))
    assert sorted(_compilable_scope(runtimes)) == ["llm", "vqvae"]
    assert _dynamic_scope(runtimes) == ["llm"]


def test_a_module_under_ddp_is_gated_off_rather_than_failing_its_wrap():
    """``janus_siglip`` ships as ``fsdp_mode: ddp``; enabling compile must not break it."""
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import module_compile_config

    module = SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=True)
    accelerator = SimpleNamespace(
        torch_compile=SimpleNamespace(enable=True, backend="inductor", mode=None, fullgraph=True, dynamic=False),
        fsdp_config=SimpleNamespace(fsdp_mode="ddp"),
    )

    assert module_compile_config(accelerator, module).enable is False


def test_compilable_scope_honours_a_module_overlay_that_turns_compile_off():
    """A per-module ``accelerator.torch_compile.enable: false`` keeps that module eager.

    Modules carry per-module ``accelerator:`` overlays and ``_validate_omni_accelerator``
    runs on each resolved copy, so an overlay is a supported way to opt a single
    module out. The composed scope must read the merged value, not only the
    top-level switch.
    """
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import _compilable_scope

    opted_in = SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=True)
    runtimes = {
        "llm": _fake_runtime(opted_in, enable=True),
        "vision": _fake_runtime(opted_in, enable=False),
    }

    assert _compilable_scope(runtimes) == ["llm"]


def test_compilable_scope_is_empty_when_every_module_overlay_turns_compile_off():
    """Nothing to compile is a valid state, not an error: the run goes eager."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        _compilable_scope,
        _dynamic_scope,
    )

    model = SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=True)
    runtimes = {"llm": _fake_runtime(model, enable=False)}

    assert _compilable_scope(runtimes) == []
    assert _dynamic_scope(runtimes) == []


def test_a_child_only_enable_is_not_vetoed_by_the_top_level_switch():
    """`enable: true` on one module compiles it even when the launcher default is off.

    The composed path used to AND the top-level switch into
    ``compile_config.enable``, which silently dropped a per-module opt-in — the
    mirror of ignoring a per-module opt-out. The scopes carry the decision, and
    the top-level switch is only the default each module's resolution starts
    from.
    """
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import (
        _compilable_scope,
        _dynamic_scope,
    )

    model = SimpleNamespace(supports_torch_compile=True, supports_torch_compile_dynamic=True)
    runtimes = {"llm": _fake_runtime(model, enable=True)}

    # This is what the composed wrap derives, and what it hands compile_decoder_blocks.
    scopes = _compilable_scope(runtimes)
    assert scopes == ["llm"]
    assert bool(scopes) is True  # `compile_config.enable` is `bool(scopes)`, not ANDed with the global flag
    assert _dynamic_scope(runtimes) == ["llm"]


def test_dynamic_scopes_reach_the_blocks_they_name():
    """`dynamic_scopes` only bites when the run-level `dynamic` is also on.

    That is why the composed wrap derives `compile_config.dynamic` from the
    scopes: copying the top-level flag would leave a module that opted into
    `dynamic` compiled static, silently.
    """
    import torch.nn as nn

    from veomni.distributed.torch_compile import CompileConfig, compile_decoder_blocks

    class Block(nn.Module):
        def forward(self, x):
            return x

    class TwoChildren(nn.Module):
        _no_split_modules = ["llm.Block", "codec.Block"]

        def __init__(self):
            super().__init__()
            self.llm = nn.Module()
            self.llm.layers = nn.ModuleList([Block()])
            self.codec = nn.Module()
            self.codec.blocks = nn.ModuleList([Block()])

    import torch

    original = torch.compile
    torch.compile = lambda fn, **_: fn
    try:
        model = TwoChildren()
        compile_decoder_blocks(
            model,
            CompileConfig(enable=True, dynamic=True),  # derived as bool(dynamic_scopes)
            scopes=["llm", "codec"],
            dynamic_scopes=["llm"],
        )
    finally:
        torch.compile = original

    assert model.llm.layers[0]._veomni_compile_config["dynamic"] is True
    assert model.codec.blocks[0]._veomni_compile_config["dynamic"] is False
