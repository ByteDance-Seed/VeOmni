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

"""Unit tests for TensorBoardTraceCallback.

These tests import the callback class directly from its source module
(bypassing the full veomni.trainer.callbacks package) to avoid triggering
heavy model-loading transitive imports that require a specific transformers
version.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import pytest


def _import_trace_callback():
    """Import trace_callback.py directly without triggering the package __init__."""
    # First, ensure the lightweight dependencies are importable.
    # Stub the Callback base class and logger since they import from veomni internals.
    base_mod_name = "veomni.trainer.callbacks.base"
    if base_mod_name not in sys.modules:
        import types

        stub = types.ModuleType(base_mod_name)

        class _StubCallback:
            def __init__(self, trainer):
                self.trainer = trainer

            def on_train_begin(self, state, **kw):
                pass

            def on_train_end(self, state, **kw):
                pass

            def on_step_begin(self, state, **kw):
                pass

            def on_step_end(self, state, **kw):
                pass

            def on_epoch_begin(self, state, **kw):
                pass

            def on_epoch_end(self, state, **kw):
                pass

        @dataclass
        class _TrainerState:
            global_step: int = 0
            epoch: int = 0

        stub.Callback = _StubCallback
        stub.TrainerState = _TrainerState
        sys.modules[base_mod_name] = stub

    # Stub veomni.distributed.parallel_state
    ps_mod_name = "veomni.distributed.parallel_state"
    if ps_mod_name not in sys.modules:
        import types

        stub = types.ModuleType(ps_mod_name)
        stub.get_parallel_state = lambda: None
        sys.modules[ps_mod_name] = stub

    # Stub veomni.utils.helper
    helper_mod_name = "veomni.utils.helper"
    if helper_mod_name not in sys.modules:
        import types

        stub = types.ModuleType(helper_mod_name)
        sys.modules[helper_mod_name] = stub
    utils_mod_name = "veomni.utils"
    if utils_mod_name not in sys.modules:
        import types

        stub = types.ModuleType(utils_mod_name)
        stub.helper = sys.modules[helper_mod_name]
        sys.modules[utils_mod_name] = stub

    # Stub veomni.utils.dist_utils
    dist_mod_name = "veomni.utils.dist_utils"
    if dist_mod_name not in sys.modules:
        import types

        stub = types.ModuleType(dist_mod_name)
        stub.all_reduce = lambda v, group=None: v
        sys.modules[dist_mod_name] = stub

    # Stub veomni.utils.logging
    logging_mod_name = "veomni.utils.logging"
    if logging_mod_name not in sys.modules:
        import types

        stub = types.ModuleType(logging_mod_name)

        class _FakeLogger:
            def info_rank0(self, msg, *a, **kw):
                pass

            def warning_rank0(self, msg, *a, **kw):
                pass

        stub.get_logger = lambda name: _FakeLogger()
        sys.modules[logging_mod_name] = stub

    # Now import the actual module
    spec = importlib.util.spec_from_file_location(
        "veomni.trainer.callbacks.trace_callback",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "veomni",
            "trainer",
            "callbacks",
            "trace_callback.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_trace_mod = _import_trace_callback()
TensorBoardTraceCallback = _trace_mod.TensorBoardTraceCallback


@dataclass
class _TBConfig:
    enable: bool = True
    save_dir: Optional[str] = None


@dataclass
class _CheckpointConfig:
    output_dir: str = "/tmp/test_output"


@dataclass
class _TrainArgs:
    global_rank: int = 0
    tensorboard: _TBConfig = field(default_factory=_TBConfig)
    checkpoint: _CheckpointConfig = field(default_factory=_CheckpointConfig)


@dataclass
class _Args:
    train: _TrainArgs = field(default_factory=_TrainArgs)


class _FakeTrainer:
    def __init__(self, args, step_env_metrics=None):
        self.args = args
        self.step_env_metrics = step_env_metrics or {}


@dataclass
class _FakeState:
    global_step: int = 1


@pytest.fixture
def tmp_tb_dir():
    d = tempfile.mkdtemp(prefix="veomni_tb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_tensorboard_callback_creates_writer(tmp_tb_dir):
    """TensorBoardTraceCallback creates a SummaryWriter on on_train_begin."""
    args = _Args(train=_TrainArgs(tensorboard=_TBConfig(enable=True, save_dir=tmp_tb_dir)))
    trainer = _FakeTrainer(args)
    cb = TensorBoardTraceCallback(trainer)

    cb.on_train_begin(_FakeState())

    assert cb.writer is not None
    assert os.path.isdir(tmp_tb_dir)

    cb.on_train_end(_FakeState())
    assert cb.writer is None


def test_tensorboard_callback_writes_scalars(tmp_tb_dir):
    """TensorBoardTraceCallback writes scalar metrics on on_step_end."""
    args = _Args(train=_TrainArgs(tensorboard=_TBConfig(enable=True, save_dir=tmp_tb_dir)))
    trainer = _FakeTrainer(args, step_env_metrics={"training/loss": 2.5, "training/lr": 1e-4})
    cb = TensorBoardTraceCallback(trainer)

    cb.on_train_begin(_FakeState(global_step=0))
    cb.on_step_end(_FakeState(global_step=1))
    cb.on_step_end(_FakeState(global_step=2))
    cb.on_train_end(_FakeState(global_step=2))

    event_files = [f for f in os.listdir(tmp_tb_dir) if "events.out.tfevents" in f]
    assert len(event_files) > 0, "Expected TensorBoard event files to be created"


def test_tensorboard_callback_disabled_non_rank0(tmp_tb_dir):
    """TensorBoardTraceCallback does nothing on non-rank-0 processes."""
    args = _Args(train=_TrainArgs(global_rank=1, tensorboard=_TBConfig(enable=True, save_dir=tmp_tb_dir)))
    trainer = _FakeTrainer(args)
    cb = TensorBoardTraceCallback(trainer)

    cb.on_train_begin(_FakeState())
    assert cb.writer is None


def test_tensorboard_callback_disabled_by_config(tmp_tb_dir):
    """TensorBoardTraceCallback does nothing when enable=False."""
    args = _Args(train=_TrainArgs(tensorboard=_TBConfig(enable=False, save_dir=tmp_tb_dir)))
    trainer = _FakeTrainer(args)
    cb = TensorBoardTraceCallback(trainer)

    cb.on_train_begin(_FakeState())
    assert cb.writer is None


def test_tensorboard_callback_default_save_dir():
    """TensorBoardTraceCallback falls back to output_dir/tensorboard when save_dir is None."""
    output_dir = tempfile.mkdtemp(prefix="veomni_tb_output_")
    try:
        args = _Args(
            train=_TrainArgs(
                tensorboard=_TBConfig(enable=True, save_dir=None),
                checkpoint=_CheckpointConfig(output_dir=output_dir),
            )
        )
        trainer = _FakeTrainer(args)
        cb = TensorBoardTraceCallback(trainer)

        cb.on_train_begin(_FakeState())
        assert cb.writer is not None

        expected_dir = os.path.join(output_dir, "tensorboard")
        assert os.path.isdir(expected_dir)

        cb.on_train_end(_FakeState())
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_tensorboard_callback_graceful_without_package(tmp_tb_dir, monkeypatch):
    """TensorBoardTraceCallback logs a warning if tensorboard is not installed."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("Mocked: tensorboard not installed")
        return real_import(name, *args, **kwargs)

    args = _Args(train=_TrainArgs(tensorboard=_TBConfig(enable=True, save_dir=tmp_tb_dir)))
    trainer = _FakeTrainer(args)
    cb = TensorBoardTraceCallback(trainer)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    cb.on_train_begin(_FakeState())
    assert cb.writer is None


def test_tensorboard_config_dataclass():
    """TensorBoardConfig dataclass has correct defaults."""
    spec = importlib.util.spec_from_file_location(
        "veomni.arguments.arguments_types",
        os.path.join(os.path.dirname(__file__), "..", "veomni", "arguments", "arguments_types.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    # Stub dependencies required by arguments_types at import time
    logging_stub_name = "veomni.utils.logging"
    if logging_stub_name not in sys.modules:
        import types

        stub = types.ModuleType(logging_stub_name)

        class _FL:
            def info_rank0(self, *a, **kw):
                pass

            def warning_rank0(self, *a, **kw):
                pass

        stub.get_logger = lambda name: _FL()
        sys.modules[logging_stub_name] = stub

    env_stub_name = "veomni.utils.env"
    if env_stub_name not in sys.modules:
        import types

        stub = types.ModuleType(env_stub_name)
        stub.get_env = lambda key, default=None: default
        sys.modules[env_stub_name] = stub

    spec.loader.exec_module(mod)

    TensorBoardConfig = mod.TensorBoardConfig
    cfg = TensorBoardConfig()
    assert cfg.enable is False
    assert cfg.save_dir is None

    # Also verify it's a field in TrainingArguments
    TrainingArguments = mod.TrainingArguments
    field_names = [f.name for f in TrainingArguments.__dataclass_fields__.values()]
    assert "tensorboard" in field_names
