from types import SimpleNamespace

from torch.profiler import ProfilerAction, schedule

from veomni.utils import helper


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
