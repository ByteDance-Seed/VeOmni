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

"""Unit tests for OmniTrainer step contexts and OmniModelRuntime forwarding (no GPU / distributed)."""

import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import psutil
import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import run_bounded_process_group
from veomni.distributed.offloading import build_activation_offloading_context, custom_save_on_cpu


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_build_activation_offloading_disabled():
    """Activation offloading returns nullcontext pair when disabled."""
    fwd, bwd = build_activation_offloading_context(enable_activation=False, enable_gradient_checkpointing=False)
    assert isinstance(fwd, nullcontext)
    assert isinstance(bwd, nullcontext)


def test_build_activation_offloading_enabled():
    """Activation offloading wraps forward in custom_save_on_cpu when enabled."""
    fwd, bwd = build_activation_offloading_context(
        enable_activation=True,
        enable_gradient_checkpointing=False,
        activation_gpu_limit=1.0,
    )
    assert isinstance(fwd, custom_save_on_cpu)
    assert isinstance(bwd, nullcontext)


def test_build_training_context_offload_flags():
    """_build_training_context wires nullcontext when offload is disabled."""
    from veomni.trainer.omni.omni_trainer import OmniTrainer

    trainer = object.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(
            accelerator=SimpleNamespace(
                offload_config=None,
                gradient_checkpointing=SimpleNamespace(enable=False),
            ),
        ),
        train=SimpleNamespace(enable_batch_invariant_mode=False),
    )
    OmniTrainer._build_training_context(trainer)
    assert isinstance(trainer.model_fwd_context, nullcontext)
    assert isinstance(trainer.model_bwd_context, nullcontext)


def test_omni_model_runtime_forwards_composed_model_surface():
    """The runtime is the single model handle: unknown attrs reach the OmniModel."""
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime

    model = SimpleNamespace(config="omni-config", modules_dict={"a": "module-a"})
    runtime = OmniModelRuntime(model)

    assert runtime.config == "omni-config"
    assert runtime.modules_dict == {"a": "module-a"}
    with pytest.raises(AttributeError):
        _ = runtime.not_a_model_attribute


def test_run_bounded_process_group_kills_sigterm_ignoring_torchrun_workers(tmp_path):
    """Workers in separate sessions that ignore SIGTERM must not survive timeout cleanup."""
    pid_file = tmp_path / "worker.pid"
    worker_script = tmp_path / "sigterm_ignoring_worker.py"
    worker_script.write_text(
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    # Emulate torchrun's exact process-group topology (launcher in its own session
    # spawning a worker with start_new_session=True) using stdlib-only Python so
    # slow torch / torch_npu import on Ascend cannot exceed the test timeout.
    launcher_script = tmp_path / "fake_torchrun_launcher.py"
    launcher_script.write_text(
        "import subprocess, sys\n"
        f"proc = subprocess.Popen([sys.executable, {str(worker_script)!r}], start_new_session=True)\n"
        "proc.wait()\n",
        encoding="utf-8",
    )
    cmd = [sys.executable, str(launcher_script)]
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded_process_group(cmd, timeout=1.0, grace_timeout=0.5)

    assert pid_file.is_file(), "worker did not start before timeout"
    worker_pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        worker_status = psutil.Process(worker_pid).status()
    except psutil.NoSuchProcess:
        worker_status = None
    assert worker_status in (None, psutil.STATUS_ZOMBIE), (
        f"worker {worker_pid} still alive with status={worker_status}"
    )
