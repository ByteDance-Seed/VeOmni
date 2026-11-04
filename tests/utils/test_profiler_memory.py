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

    def step(self):
        self.step_num += 1
        self.current_action = self._schedule(self.step_num)

    def stop(self):
        pass


def _recorded_steps(monkeypatch, start_step: int, end_step: int, train_steps: int) -> list[int]:
    calls = []
    memory = SimpleNamespace(_record_memory_history=lambda enabled="all": calls.append(enabled))
    monkeypatch.setattr(helper, "get_torch_device", lambda: SimpleNamespace(memory=memory))

    # Same arithmetic as ``create_profiler``.
    warmup = 0 if start_step == 1 else 1
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


def test_memory_history_covers_only_the_profiled_window(monkeypatch):
    # Warmup step 19 plus the active step 20; nothing from the 18 steps before.
    assert _recorded_steps(monkeypatch, start_step=20, end_step=21, train_steps=30) == [19, 20]


def test_memory_history_from_the_first_step_when_the_window_starts_there(monkeypatch):
    assert _recorded_steps(monkeypatch, start_step=1, end_step=3, train_steps=5) == [1, 2]


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
