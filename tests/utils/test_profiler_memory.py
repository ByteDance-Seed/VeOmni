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

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerAction, schedule
from torch.utils.checkpoint import checkpoint

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils import helper
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_id, get_device_type, synchronize
from veomni.utils.helper import create_profiler


class _ScheduledProfiler:
    """The slice of ``torch.profiler.profile`` the wrapper reads: a schedule-driven action."""

    def __init__(self, wait: int, warmup: int, active: int):
        self._schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
        self.step_num = 0
        self.current_action = ProfilerAction.NONE

    def start(self):
        self.current_action = self._schedule(self.step_num)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def step(self):
        self.step_num += 1
        self.current_action = self._schedule(self.step_num)

    def stop(self):
        pass

    def export_chrome_trace(self, path: str) -> str:
        return f"exported:{path}"


def _recorded_steps(monkeypatch, start_step: int, end_step: int, train_steps: int, *, warmup: int = 0) -> list[int]:
    calls = []
    memory = SimpleNamespace(_record_memory_history=lambda enabled="all": calls.append(enabled))
    monkeypatch.setattr(helper, "get_torch_device", lambda: SimpleNamespace(memory=memory))

    profiler = helper.ProfilerWithMem(
        _ScheduledProfiler(wait=start_step - warmup - 1, warmup=warmup, active=end_step - start_step)
    )
    recording = []
    profiler.start()
    for step in range(1, train_steps + 1):
        if profiler._recording:
            recording.append(step)
        profiler.step()
    profiler.stop()
    assert calls[-1] is None, "history must be off once the run is over"
    return recording


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_memory_history_covers_only_the_profiled_window(monkeypatch):
    # With warmup=1: warmup step 19 plus active step 20; nothing from the 18 steps before.
    assert _recorded_steps(monkeypatch, start_step=20, end_step=21, train_steps=30, warmup=1) == [19, 20]
    # With warmup=0 (create_profiler default): only active step 20.
    assert _recorded_steps(monkeypatch, start_step=20, end_step=21, train_steps=30, warmup=0) == [20]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_memory_history_from_the_first_step_when_the_window_starts_there(monkeypatch):
    assert _recorded_steps(monkeypatch, start_step=1, end_step=3, train_steps=5) == [1, 2]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_profiler_with_mem_context_manager_returns_wrapper_and_forwards_attrs(monkeypatch):
    """``with ProfilerWithMem(...) as p:`` returns the wrapper so ``p.step()`` follows the schedule."""
    calls = []
    memory = SimpleNamespace(_record_memory_history=lambda enabled="all": calls.append(enabled))
    monkeypatch.setattr(helper, "get_torch_device", lambda: SimpleNamespace(memory=memory))

    wrapper = helper.ProfilerWithMem(_ScheduledProfiler(wait=1, warmup=0, active=1))
    recorded = []
    with wrapper as p:
        assert p is wrapper
        assert p.export_chrome_trace("trace.json") == "exported:trace.json"
        for step in range(1, 5):
            if p._recording:
                recorded.append(step)
            p.step()

    assert recorded == [2]
    assert wrapper._recording is False
    assert calls[-1] is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_create_profiler_skips_memory_dump_when_profile_memory_false(monkeypatch, tmp_path):
    captured = {}
    dump_calls = []

    def fake_schedule(*, wait, warmup, active, repeat):
        captured["schedule"] = (wait, warmup, active, repeat)
        return "sched"

    def fake_profile(**kwargs):
        captured["handler"] = kwargs["on_trace_ready"]
        return SimpleNamespace(export_chrome_trace=lambda path: None)

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(helper, "IS_MLU_AVAILABLE", False)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", "")
    monkeypatch.setattr(
        helper,
        "get_torch_device",
        lambda: SimpleNamespace(memory=SimpleNamespace(_dump_snapshot=lambda p: dump_calls.append(p))),
    )
    monkeypatch.setattr(torch.profiler, "schedule", fake_schedule)
    monkeypatch.setattr(torch.profiler, "profile", fake_profile)

    prof = create_profiler(
        start_step=5,
        end_step=7,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
    )
    assert captured["schedule"] == (4, 0, 2, 1)
    assert not isinstance(prof, helper.ProfilerWithMem)
    captured["handler"](prof)
    assert dump_calls == []


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA memory history requires a GPU")
def test_profiler_memory_snapshot_keeps_live_allocation_stacks(tmp_path):
    profiler = create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=False,
        global_rank=0,
    )

    profiler.start()
    try:
        device = get_device_type()
        values = torch.randn(128, 64, device=device, requires_grad=True)
        indices = torch.arange(128, device=device) % 16
        forward_calls = 0

        def recomputed_forward(inputs):
            nonlocal forward_calls
            forward_calls += 1
            output = torch.zeros_like(inputs)
            output.index_add_(0, indices, inputs)
            return output.square()

        checkpoint(recomputed_forward, values, use_reentrant=False).mean().backward()
        assert forward_calls > 1
        synchronize()
        profiler.step()
    finally:
        profiler.stop()

    with next(tmp_path.glob("*.pkl")).open("rb") as file:
        snapshot = pickle.load(file)

    live_blocks = [
        block
        for segment in snapshot["segments"]
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    ]
    assert any(block["frames"] for block in live_blocks)

    # Free-event traceback capture can fail during checkpoint recomputation
    # and is unnecessary for locating the tensors still occupying memory.
    events = snapshot["device_traces"][get_device_id()]
    assert events
    assert all(not event.get("frames") for event in events)
    assert list(tmp_path.glob("*.pt.trace.json.gz"))
