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

from types import SimpleNamespace

import pytest
import torch.nn as nn

from veomni.distributed import parallel_state
from veomni.distributed.parallel_state import ParallelState
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.models.seed_omni.accelerator.module_runtime import ModuleRuntime


def _unbuilt(model: nn.Module | None = None, **args_fields) -> ModuleRuntime:
    """A ModuleRuntime with its fields set but no build run."""
    runtime = ModuleRuntime.__new__(ModuleRuntime)
    runtime.model = model
    runtime.model_name = "vision_encoder"
    runtime.args = SimpleNamespace(model_path="/tmp/hf-model", lora_config=None, **args_fields)
    runtime.train = None
    return runtime


def test_module_runtime_is_a_model_runtime():
    assert issubclass(ModuleRuntime, VeOmniModelRuntime)


def test_omni_runtime_arguments_are_model_arguments():
    from veomni.arguments import ModelArguments, OmniModelRuntimeArguments, OmniModuleRuntimeArguments

    assert issubclass(OmniModuleRuntimeArguments, ModelArguments)
    assert issubclass(OmniModelRuntimeArguments, ModelArguments)


def test_module_checkpoint_manager_is_a_model_checkpoint_manager():
    from veomni.models.checkpoint_manager import ModelCheckpointManager
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


def test_build_model_reads_the_modules_own_directory(monkeypatch):
    """``config_path`` is inherited from the composed model and points at the omni
    checkpoint root, whose ``config.json`` is the OmniConfig — not this module."""
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
            fsdp_config=SimpleNamespace(mixed_precision=SimpleNamespace(enable=False)),
        ),
    )
    runtime._build_model()

    assert captured["config_path"] == "/tmp/hf-model"


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
    assert runtime.checkpoint_subfolder == "vision_encoder"


def test_a_trainable_module_does_get_a_checkpoint_manager(monkeypatch):
    """The frozen carve-out must be a carve-out, not the only path."""
    built = []
    monkeypatch.setattr(
        "veomni.models.seed_omni.accelerator.module_runtime.OmniModuleCheckpointManager",
        lambda runtime: built.append(runtime) or "manager",
    )

    runtime = _unbuilt(nn.Linear(2, 2))
    runtime.build_checkpoint()

    assert runtime.checkpoint == "manager"
    assert built == [runtime]


def test_a_module_the_lora_config_missed_stays_frozen_instead_of_failing():
    """A composed ``lora_config`` reaches every module, so "no targets here" is
    how it says which model to adapt — only the composer can call it an error."""
    runtime = _unbuilt(nn.Linear(2, 2))

    runtime.on_lora_matched_nothing()  # must not raise


def test_a_module_does_not_write_the_jobs_model_assets():
    with pytest.raises(NotImplementedError, match="collect_hf_export_assets"):
        _unbuilt(nn.Linear(2, 2)).save_model_assets()


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


def test_a_direct_forward_still_works_for_an_unregistered_eager_module():
    """An eager-inference module never runs ``setup``, so it has no state to
    enter — the same carve-out ``OmniModelRuntime.module_context`` makes."""
    assert not parallel_state.is_parallel_state_registered("vision_encoder")

    assert _unbuilt(_RecordAmbientState())(7) == 7
