from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.arguments_types import ProfileConfig
from veomni.arguments.parser import parse_args
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
    ("analysis_mode", "expected_kwargs"),
    [
        ("offline", {"analyse_flag": False}),
        ("async", {"analyse_flag": True, "async_mode": True}),
    ],
)
def test_npu_analysis_mode_controls_trace_handler(monkeypatch, tmp_path, analysis_mode, expected_kwargs):
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
        npu_analysis_mode=analysis_mode,
    )

    assert fake_profiler.trace_handler_calls == [(str(tmp_path), expected_kwargs)]


def test_npu_analysis_mode_defaults_to_offline():
    assert ProfileConfig().npu_analysis_mode == "offline"


def test_npu_analysis_mode_rejects_invalid_yaml_value():
    with pytest.raises(ValueError, match="npu_analysis_mode"):
        ProfileConfig(npu_analysis_mode="sync")


def test_legacy_npu_offline_true_maps_to_offline(monkeypatch):
    warnings = []
    monkeypatch.setattr("veomni.arguments.arguments_types.logger.warning", warnings.append)

    config = ProfileConfig(npu_offline_analysis=True)

    assert config.npu_analysis_mode == "offline"
    assert any("deprecated" in warning for warning in warnings)


def test_legacy_npu_offline_false_requires_explicit_safe_mode():
    with pytest.raises(ValueError, match="removed synchronous online analysis"):
        ProfileConfig(enable=True, npu_offline_analysis=False)


def test_disabled_profile_loads_legacy_saved_false_value(monkeypatch, tmp_path):
    @dataclass
    class _TrainConfig:
        profile: ProfileConfig = field(default_factory=ProfileConfig)

    @dataclass
    class _SavedConfig:
        train: _TrainConfig = field(default_factory=_TrainConfig)

    config_path = tmp_path / "legacy_veomni_cli.yaml"
    config_path.write_text(
        "train:\n  profile:\n    enable: false\n    npu_offline_analysis: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["test", str(config_path)])

    parsed = parse_args(_SavedConfig)

    assert parsed.train.profile.enable is False
    assert parsed.train.profile.npu_offline_analysis is False
    assert parsed.train.profile.npu_analysis_mode == "offline"


def test_create_profiler_accepts_legacy_offline_true(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)

    with pytest.deprecated_call(match="npu_offline_analysis"):
        helper.create_profiler(
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

    assert fake_profiler.trace_handler_calls == [(str(tmp_path), {"analyse_flag": False})]


@pytest.mark.parametrize("analysis_mode", ["offline", "async"])
def test_npu_analysis_does_not_record_unused_memory_history(monkeypatch, tmp_path, analysis_mode):
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
        npu_analysis_mode=analysis_mode,
    )

    assert not isinstance(profiler, helper.ProfilerWithMem)


def test_npu_offline_spawns_async_upload_sidecar(monkeypatch, tmp_path):
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
        lambda *args, **kwargs: (sidecar_calls.append((args, kwargs)), SimpleNamespace(pid=123))[1],
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
        npu_analysis_mode="offline",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == [
        (
            (str(raw_dir),),
            {"copy_to": None, "analyse": True, "upload_cmd": "upload-trace", "merlin_upload": False},
        )
    ]
    assert [sidecar.pid for sidecar in profiler._veomni_npu_sidecars] == [123]


def test_npu_sidecar_log_setup_failure_is_nonfatal(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(helper.os, "makedirs", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn"))
    monkeypatch.setattr(helper.logger, "warning", warnings.append)

    proc = helper.spawn_npu_offline_sidecar(str(tmp_path / "rank0_ascend_pt"), analyse=True)

    assert proc is None
    assert any("disk full" in warning for warning in warnings)


def test_wait_npu_profile_sidecars_reports_completion_and_timeout(monkeypatch):
    logs = []
    warnings = []

    class _Completed:
        pid = 101

        def wait(self, timeout):
            return 0

    class _Pending:
        pid = 102

        def wait(self, timeout):
            raise helper.subprocess.TimeoutExpired(cmd="sidecar", timeout=timeout)

    profiler = SimpleNamespace(_veomni_npu_sidecars=[_Completed(), _Pending()])
    monkeypatch.setattr(helper.logger, "info", logs.append)
    monkeypatch.setattr(helper.logger, "warning", warnings.append)

    helper.wait_npu_profile_sidecars(profiler, timeout_seconds=0.01)

    assert any("pid=101 status=completed" in message for message in logs)
    assert any("pid=102" in warning and "still running" in warning for warning in warnings)


def test_npu_offline_defers_hdfs_copy_to_sidecar(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    sidecar_calls = []
    copies = []
    trace_dir = "hdfs://haruna/profile"
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", None)
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
        lambda *args, **kwargs: (sidecar_calls.append((args, kwargs)), SimpleNamespace(pid=123))[1],
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
        npu_analysis_mode="offline",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert copies == []
    assert sidecar_calls == [
        (
            (str(raw_dir),),
            {"copy_to": trace_dir, "analyse": False, "upload_cmd": None, "merlin_upload": False},
        )
    ]


def test_npu_offline_sidecar_opt_out_never_falls_back_to_sync_hdfs_copy(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    warnings = []
    sidecar_calls = []
    copies = []
    trace_dir = "hdfs://haruna/profile"
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", None)
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
        npu_analysis_mode="offline",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == []
    assert copies == []
    assert any("raw capture remains" in warning for warning in warnings)


def test_npu_offline_spawns_merlin_upload_sidecar(monkeypatch, tmp_path):
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
        lambda *args, **kwargs: (sidecar_calls.append((args, kwargs)), SimpleNamespace(pid=123))[1],
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
        npu_analysis_mode="offline",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == [
        (
            (str(raw_dir),),
            {"copy_to": None, "analyse": True, "upload_cmd": None, "merlin_upload": True},
        )
    ]


def test_npu_offline_sidecar_failure_preserves_raw_without_sync_fallback(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()
    warnings = []

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", "upload-trace")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", None)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper, "spawn_npu_offline_sidecar", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "copy", lambda *args, **kwargs: pytest.fail("must not sync-copy"))
    monkeypatch.setattr(helper.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not sync-upload"))
    monkeypatch.setattr(helper.logger, "warning", warnings.append)

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_analysis_mode="offline",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert raw_dir.is_dir()
    assert any(str(raw_dir) in warning and "no synchronous fallback" in warning for warning in warnings)


def test_npu_async_rejects_hdfs_trace_dir(monkeypatch):
    fake_profiler = _FakeNpuProfiler()
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)

    with pytest.raises(ValueError, match="pod-local"):
        helper.create_profiler(
            start_step=1,
            end_step=2,
            trace_dir="hdfs://haruna/profile",
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            global_rank=0,
            npu_analysis_mode="async",
        )


def test_npu_async_does_not_race_background_analysis_with_upload(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()
    sidecar_calls = []
    warnings = []

    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", "upload-trace")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", "1")
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", "1")
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(
        helper, "spawn_npu_offline_sidecar", lambda *args, **kwargs: sidecar_calls.append((args, kwargs))
    )
    monkeypatch.setattr(helper, "copy", lambda *args, **kwargs: pytest.fail("must not copy before async analysis"))
    monkeypatch.setattr(helper.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not upload early"))
    monkeypatch.setattr(helper.logger, "warning", warnings.append)

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_analysis_mode="async",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert sidecar_calls == []
    assert any("async analysis" in warning for warning in warnings)


def test_npu_async_submission_failure_preserves_raw_and_does_not_fail_training(monkeypatch, tmp_path):
    class _FailingAsyncNpuProfiler(_FakeNpuProfiler):
        def tensorboard_trace_handler(self, trace_dir, **kwargs):
            self.trace_handler_calls.append((trace_dir, kwargs))

            def handler(profiler):
                raise RuntimeError("process pool unavailable")

            return handler

    fake_profiler = _FailingAsyncNpuProfiler()
    raw_dir = tmp_path / "rank0_ascend_pt"
    raw_dir.mkdir()
    warnings = []
    logs = []
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "VEOMNI_UPLOAD_CMD", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_POSTPROCESS", None)
    monkeypatch.setattr(helper, "VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD", None)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper.logger, "warning", warnings.append)
    monkeypatch.setattr(helper.logger, "info", logs.append)

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_analysis_mode="async",
    )
    profiler.on_trace_ready(SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(raw_dir))))

    assert raw_dir.is_dir()
    assert any("training will continue" in warning and str(raw_dir) in warning for warning in warnings)
    assert any("status=analysis_submit_failed" in message for message in logs)


def test_npu_async_falls_back_to_offline_in_daemon_process(monkeypatch, tmp_path):
    fake_profiler = _FakeNpuProfiler()
    warnings = []
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper.multiprocessing, "current_process", lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(helper.logger, "warning", warnings.append)

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_analysis_mode="async",
    )

    assert fake_profiler.trace_handler_calls == [(str(tmp_path), {"analyse_flag": False})]
    assert profiler._veomni_npu_analysis_mode == "offline"
    assert any("daemon" in warning for warning in warnings)


def test_npu_async_falls_back_when_torch_npu_does_not_support_it(monkeypatch, tmp_path):
    class _LegacyNpuProfiler(_FakeNpuProfiler):
        def tensorboard_trace_handler(self, trace_dir, analyse_flag=True):
            self.trace_handler_calls.append((trace_dir, {"analyse_flag": analyse_flag}))
            return lambda profiler: None

    fake_profiler = _LegacyNpuProfiler()
    warnings = []
    monkeypatch.setattr(helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(helper, "IS_CUDA_AVAILABLE", False)
    monkeypatch.setattr(helper, "torch_npu", SimpleNamespace(profiler=fake_profiler), raising=False)
    monkeypatch.setattr(helper.logger, "warning", warnings.append)

    profiler = helper.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        global_rank=0,
        npu_analysis_mode="async",
    )

    assert fake_profiler.trace_handler_calls == [(str(tmp_path), {"analyse_flag": False})]
    assert profiler._veomni_npu_analysis_mode == "offline"
    assert any("does not support" in warning for warning in warnings)


@pytest.mark.parametrize("npu_analysis_mode", ["offline", "async"])
@pytest.mark.parametrize(
    ("this_rank", "expected_events"),
    [
        (True, ["barrier", "step", "barrier"]),
        (False, ["barrier", "barrier"]),
    ],
)
def test_npu_profile_synchronizes_finalization(monkeypatch, npu_analysis_mode, this_rank, expected_events):
    events = []
    profile_config = SimpleNamespace(
        enable=True,
        npu_analysis_mode=npu_analysis_mode,
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


def test_npu_profile_finalize_error_still_releases_post_barrier(monkeypatch):
    events = []
    profile_config = SimpleNamespace(enable=True, npu_analysis_mode="offline", end_step=6, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(profile=profile_config)))
    callback._profile_active = True

    def fail_step():
        events.append("step")
        raise RuntimeError("raw dump failed")

    callback.profiler = SimpleNamespace(step=fail_step, stop=lambda: None)
    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))

    with pytest.raises(RuntimeError, match="raw dump failed"):
        callback.on_step_end(TrainerState(global_step=5))

    assert events == ["barrier", "step", "barrier"]


def test_npu_profile_logs_full_step_and_finalize_breakdown(monkeypatch):
    events = []
    logs = []
    profile_config = SimpleNamespace(
        enable=True,
        npu_analysis_mode="async",
        end_step=6,
        this_rank=True,
    )
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(profile=profile_config, global_rank=0))
    )
    callback._profile_active = True
    callback._profile_timing_start_step = 5
    callback.profiler = SimpleNamespace(
        _veomni_npu_analysis_mode="async",
        step=lambda: events.append("step"),
        stop=lambda: events.append("stop"),
    )
    clock = iter([1.0, 2.0, 2.1, 3.0, 3.2, 4.0, 4.3, 5.0])

    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(trace_callback.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(trace_callback.logger, "info", logs.append)
    monkeypatch.setattr(trace_callback.logger, "info_rank0", logs.append)

    callback.on_step_begin(TrainerState(global_step=5))
    callback.on_step_end(TrainerState(global_step=5))

    assert events == ["barrier", "step", "barrier"]
    assert len(logs) == 1
    assert "mode=async" in logs[0]
    assert "total_seconds=4.000000" in logs[0]
    assert "pre_barrier_seconds=0.100000" in logs[0]
    assert "profiler_step_seconds=0.200000" in logs[0]
    assert "post_barrier_seconds=0.300000" in logs[0]


def test_npu_profile_timing_skips_steps_outside_profile_window(monkeypatch):
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(
                profile=SimpleNamespace(enable=True, npu_analysis_mode="offline", end_step=6),
                global_rank=0,
            )
        )
    )
    callback.profiler = None
    callback._profile_active = True
    callback._profile_timing_start_step = 4
    callback._step_started_at = None
    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)

    callback.on_step_begin(TrainerState(global_step=3))
    assert callback._step_started_at is None

    callback.on_step_begin(TrainerState(global_step=4))
    assert callback._step_started_at is not None

    callback._step_started_at = None
    callback.on_step_begin(TrainerState(global_step=7))
    assert callback._step_started_at is None


def test_npu_profile_stops_without_finalize_barrier_at_end_step(monkeypatch):
    events = []
    profile_config = SimpleNamespace(enable=True, npu_analysis_mode="async", end_step=6, this_rank=True)
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
    profile_config = SimpleNamespace(enable=True, npu_analysis_mode="offline", end_step=6, this_rank=True)
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
    profile_config = SimpleNamespace(enable=True, npu_analysis_mode="async", end_step=6, this_rank=True)
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


def test_npu_profile_train_end_stops_early_window_and_waits_for_sidecar(monkeypatch):
    events = []
    logs = []
    profile_config = SimpleNamespace(enable=True, npu_analysis_mode="offline", end_step=20, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(profile=profile_config, global_rank=0))
    )
    callback._profile_active = True
    callback._profiler_stopped = False
    callback.profiler = SimpleNamespace(
        _veomni_npu_analysis_mode="offline",
        stop=lambda: events.append("stop"),
    )
    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.helper, "wait_npu_profile_sidecars", lambda profiler: events.append("wait"))
    monkeypatch.setattr(trace_callback.dist, "is_available", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trace_callback.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(trace_callback.logger, "info_rank0", logs.append)

    callback.on_train_end(TrainerState(global_step=10))

    assert events == ["barrier", "stop", "barrier", "wait"]
    assert callback._profiler_stopped is True
    assert callback._profile_active is False
    assert any("NPU_PROFILE_TRAIN_END mode=offline step=10" in message for message in logs)


def test_npu_profile_train_end_does_not_stop_twice(monkeypatch):
    events = []
    profile_config = SimpleNamespace(enable=True, npu_analysis_mode="async", end_step=6, this_rank=True)
    callback = object.__new__(trace_callback.ProfileTraceCallback)
    callback.trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(profile=profile_config, global_rank=0))
    )
    callback._profile_active = False
    callback._profiler_stopped = True
    callback.profiler = SimpleNamespace(
        _veomni_npu_analysis_mode="async",
        stop=lambda: pytest.fail("profiler already stopped"),
    )
    monkeypatch.setattr(trace_callback.helper, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(trace_callback.helper, "wait_npu_profile_sidecars", lambda profiler: events.append("wait"))
    monkeypatch.setattr(trace_callback.logger, "info_rank0", lambda message: None)

    callback.on_train_end(TrainerState(global_step=30))

    assert events == ["wait"]


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
        npu_analysis_mode="offline",
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
