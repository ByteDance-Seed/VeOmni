from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.arguments_types import ProfileConfig
from veomni.trainer.callbacks import trace_callback
from veomni.trainer.callbacks.base import TrainerState
from veomni.utils import helper


class _FakeNpuProfiler:
    class ProfilerActivity:
        CPU = "cpu"
        NPU = "npu"

    class AiCMetrics:
        PipeUtilization = "pipe_utilization"

    class ProfilerLevel:
        Level1 = "level1"

    def __init__(self):
        self.trace_handler_calls = []

    def tensorboard_trace_handler(self, trace_dir, **kwargs):
        self.trace_handler_calls.append((trace_dir, kwargs))
        return lambda profiler: None

    def _ExperimentalConfig(self, **kwargs):
        return kwargs

    def schedule(self, **kwargs):
        return kwargs

    def profile(self, **kwargs):
        return SimpleNamespace(**kwargs)


@pytest.mark.parametrize(
    ("offline_analysis", "expected_kwargs"),
    [(False, {}), (True, {"analyse_flag": False})],
)
def test_npu_offline_analysis_controls_trace_handler(monkeypatch, tmp_path, offline_analysis, expected_kwargs):
    fake_profiler = _FakeNpuProfiler()
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)

    helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_offline_analysis=offline_analysis,
    )

    assert fake_profiler.trace_handler_calls == [(str(tmp_path), expected_kwargs)]


def test_npu_offline_analysis_is_disabled_by_default():
    assert ProfileConfig().npu_offline_analysis is False


def test_npu_offline_analysis_does_not_record_unused_memory_history(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_offline_analysis=True,
    )

    assert not isinstance(profiler, helper.ProfilerWithMem)


def test_npu_offline_analysis_spawns_async_upload_sidecar(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    sidecar_calls = []
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", "upload-trace")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", None)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper, "get_torch_device", lambda: pytest.fail("offline mode must not dump memory snapshots"))
    monkeypatch.setattr(
        helper,
        "spawn_npu_offline_sidecar",
        lambda *args, **kwargs: sidecar_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(helper.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not sync-upload"))

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_offline_analysis=True,
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == [
        (
            (str(raw_dir),),
            {"copy_to": None, "analyse": True, "upload_cmd": "upload-trace", "merlin_upload": False},
        )
    ]


def test_npu_offline_analysis_defers_hdfs_copy_to_sidecar(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    sidecar_calls = []
    copies = []
    trace_dir = "hdfs://haruna/profile"
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", "upload-trace")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", None)
    monkeypatch.setattr(helper, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper.hdfs_io, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "copy", lambda source, target: (copies.append((source, target)), True)[1])
    monkeypatch.setattr(helper, "get_torch_device", lambda: pytest.fail("offline mode must not dump memory snapshots"))
    monkeypatch.setattr(
        helper,
        "spawn_npu_offline_sidecar",
        lambda *args, **kwargs: sidecar_calls.append((args, kwargs)),
    )

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=trace_dir,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_offline_analysis=True,
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert copies == []
    assert sidecar_calls == [
        (
            (str(raw_dir),),
            {"copy_to": trace_dir, "analyse": True, "upload_cmd": "upload-trace", "merlin_upload": False},
        )
    ]


def test_npu_offline_analysis_can_opt_out_of_sidecar(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    warnings = []
    sidecar_calls = []
    copies = []
    trace_dir = "hdfs://haruna/profile"
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", "upload-trace")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", "0")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", None)
    monkeypatch.setattr(helper, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper.hdfs_io, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "copy", lambda source, target: (copies.append((source, target)), True)[1])
    monkeypatch.setattr(helper, "get_torch_device", lambda: pytest.fail("offline mode must not dump memory snapshots"))
    monkeypatch.setattr(helper.logger, "warning", warnings.append)
    monkeypatch.setattr(
        helper,
        "spawn_npu_offline_sidecar",
        lambda *args, **kwargs: sidecar_calls.append((args, kwargs)),
    )

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=trace_dir,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_offline_analysis=True,
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == []
    assert copies == [(str(raw_dir), trace_dir)]
    assert any("VEOMNI_UPLOAD_CMD is skipped" in warning for warning in warnings)


def test_npu_offline_analysis_spawns_merlin_upload_sidecar(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    sidecar_calls = []
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", "1")
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper, "get_torch_device", lambda: pytest.fail("offline mode must not dump memory snapshots"))
    monkeypatch.setattr(
        helper,
        "spawn_npu_offline_sidecar",
        lambda *args, **kwargs: sidecar_calls.append((args, kwargs)),
    )

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_offline_analysis=True,
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == [
        (
            (str(raw_dir),),
            {"copy_to": None, "analyse": True, "upload_cmd": None, "merlin_upload": True},
        )
    ]


@pytest.mark.parametrize("npu_offline_analysis", [True, False])
@pytest.mark.parametrize(
    ("this_rank", "expected_events"),
    [
        (True, ["barrier", "step", "barrier"]),
        (False, ["barrier", "barrier"]),
    ],
)
def test_npu_profile_synchronizes_finalization(monkeypatch, npu_offline_analysis, this_rank, expected_events):
    events = []
    profile_config = SimpleNamespace(
        enable=True,
        npu_offline_analysis=npu_offline_analysis,
        end_step=6,
        this_rank=this_rank,
    )
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(profile=profile_config)))
    callback._profile_active = True
    callback.profiler = (
        SimpleNamespace(
            step=lambda: events.append("step"),
            stop=lambda: events.append("stop"),
        )
        if this_rank
        else None
    )

    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))

    callback.on_step_end(TrainerState(global_step=5))

    assert events == expected_events


def test_npu_profile_stops_without_finalize_barrier_at_end_step(monkeypatch):
    events = []
    profile_config = SimpleNamespace(enable=True, npu_offline_analysis=False, end_step=6, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(profile=profile_config)))
    callback._profile_active = True
    callback.profiler = SimpleNamespace(step=lambda: events.append("step"), stop=lambda: events.append("stop"))

    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))

    callback.on_step_end(TrainerState(global_step=6))

    assert events == ["step", "stop"]


def test_npu_profile_does_not_synchronize_before_finalize_step(monkeypatch):
    events = []
    profile_config = SimpleNamespace(enable=True, npu_offline_analysis=True, end_step=6, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(profile=profile_config)))
    callback._profile_active = True
    callback.profiler = SimpleNamespace(step=lambda: events.append("step"), stop=lambda: events.append("stop"))

    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))

    callback.on_step_end(TrainerState(global_step=4))

    assert events == ["step"]


def test_cuda_profile_does_not_use_npu_finalize_barrier(monkeypatch):
    events = []
    profile_config = SimpleNamespace(enable=True, npu_offline_analysis=False, end_step=6, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(profile=profile_config)))
    callback._profile_active = True
    callback.profiler = SimpleNamespace(step=lambda: events.append("step"), stop=lambda: events.append("stop"))

    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))

    callback.on_step_end(TrainerState(global_step=5))

    assert events == ["step"]


def test_profile_schedule_finalizes_at_end_step_minus_one():
    handler_steps = []
    current_global_step = [0]
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=1),
        on_trace_ready=lambda _: handler_steps.append(current_global_step[0]),
        acc_events=True,
    )
    profiler.start()
    for step in range(1, 6):
        current_global_step[0] = step
        profiler.step()
    profiler.stop()

    # create_profiler(start_step=5, end_step=6) builds wait=3,
    # warmup=1, active=1. Exiting RECORD_AND_SAVE invokes the handler in
    # profiler.step() at global step 5, i.e. end_step - 1.
    assert handler_steps == [5]


def test_profile_callback_rebases_absolute_steps_after_resume(monkeypatch, tmp_path):
    create_calls = []
    profiler = SimpleNamespace(start=lambda: None)
    profile_config = SimpleNamespace(
        enable=True,
        start_step=5,
        end_step=6,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        npu_offline_analysis=True,
        this_rank=True,
    )
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(profile=profile_config, global_rank=0))
    )

    monkeypatch.setattr(
        trace_callback.helper,
        "create_profiler",
        lambda **kwargs: (create_calls.append(kwargs), profiler)[1],
    )

    callback.on_train_begin(TrainerState(global_step=4))

    assert callback._profile_active is True
    assert create_calls[0]["start_step"] == 1
    assert create_calls[0]["end_step"] == 2


def test_profile_callback_skips_elapsed_window(monkeypatch):
    warnings = []
    profile_config = SimpleNamespace(enable=True, start_step=5, end_step=6, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(profile=profile_config)))

    monkeypatch.setattr(trace_callback.logger, "warning_rank0", warnings.append)
    monkeypatch.setattr(trace_callback.helper, "create_profiler", lambda **kwargs: pytest.fail("window elapsed"))

    callback.on_train_begin(TrainerState(global_step=5))

    assert callback._profile_active is False
    assert callback.profiler is None
    assert warnings == ["Profiling window [5, 6) has no remaining steps after global step 5; profiling is skipped."]
