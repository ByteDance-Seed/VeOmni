"""DCP file buckets must bound copy-ahead without changing stored tensors."""

import errno
import os
import queue
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.planner import WriteItemType

from scripts.ci.accelerator.launch_plan import LaunchSpec, RunnerCapabilities
from scripts.ci.accelerator.runtime.devices import child_environment
from veomni.checkpoint.bounded_writer import BoundedFileSystemWriter, _buckets


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _item(elements=None, dtype=torch.float32):
    if elements is None:
        return SimpleNamespace(type=WriteItemType.BYTE_IO, tensor_data=None)
    return SimpleNamespace(
        type=WriteItemType.TENSOR,
        tensor_data=SimpleNamespace(
            size=(elements * 16,), chunk=SimpleNamespace(sizes=(elements,)), properties=SimpleNamespace(dtype=dtype)
        ),
    )


def test_bucket_limits_and_oversized_or_opaque_items():
    items = [_item(4), _item(4), _item(20), _item(), _item(4), _item(4), _item(4)]
    buckets = list(_buckets(items, 32))
    assert buckets == [items[0:2], items[2:3], items[3:6], items[6:7]]
    with pytest.raises(ValueError, match="positive"):
        list(_buckets(items, 0))


def test_bucketed_dcp_roundtrip_is_bitwise_exact(tmp_path):
    source = {f"parameter_{i}": torch.arange(16, dtype=torch.float32) + i / 100 for i in range(8)}
    source["bf16"] = torch.arange(16, dtype=torch.bfloat16)
    source["byte_payload"] = "opaque payload"
    writer = BoundedFileSystemWriter(tmp_path, max_bytes=128)
    dcp.save(source, storage_writer=writer, no_dist=True)
    assert len(list(tmp_path.glob("*.distcp"))) >= 5
    metadata_before = (tmp_path / ".metadata").read_bytes()
    with pytest.raises(CheckpointException, match="Checkpoint already exists"):
        dcp.save(source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128), no_dist=True)
    assert (tmp_path / ".metadata").read_bytes() == metadata_before
    restored = {name: torch.empty_like(value) if torch.is_tensor(value) else "" for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    for name, expected in source.items():
        if torch.is_tensor(expected):
            assert torch.equal(restored[name], expected)
        else:
            assert restored[name] == expected


def test_bucketed_dcp_propagates_write_failure(tmp_path, monkeypatch):
    writer = BoundedFileSystemWriter(tmp_path, max_bytes=16)

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(writer.fs, "create_stream", fail)
    with pytest.raises(CheckpointException, match="CheckpointException|injected"):
        dcp.save({"x": torch.ones(16)}, storage_writer=writer, no_dist=True)
    assert not (tmp_path / ".metadata").exists()

    class EmptyOnSecondTensor(DefaultSavePlanner):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def resolve_data(self, item):
            self.seen += 1
            if self.seen == 2:
                raise queue.Empty("injected planner failure")
            return super().resolve_data(item)

    # Exercise failures both after a complete file and midway through one file.
    for bucket_bytes in (64, 128):
        partial = tmp_path / f"partial-{bucket_bytes}"
        with pytest.raises(CheckpointException, match="Incomplete DCP write"):
            dcp.save(
                {"x": torch.ones(16), "y": torch.ones(16)},
                storage_writer=BoundedFileSystemWriter(partial, max_bytes=bucket_bytes),
                planner=EmptyOnSecondTensor(),
                no_dist=True,
            )
        assert list(partial.glob("*.distcp"))
        assert not (partial / ".metadata").exists()


def test_bucketed_writer_releases_previous_temporary_storages(tmp_path):
    class TrackingPlanner(DefaultSavePlanner):
        def __init__(self):
            super().__init__()
            self.live = 0
            self.peak = 0

        def resolve_data(self, item):
            value = super().resolve_data(item)
            if not torch.is_tensor(value):
                return value
            value = value.clone()
            self.live += 1
            self.peak = max(self.peak, self.live)
            storage = value.untyped_storage()
            weakref.finalize(storage, self._release)
            # The Ascend image retains temporary storages through cycles in
            # the serialization path. Reproduce this even on other runtimes.
            cycle = {"storage": storage}
            cycle["self"] = cycle
            return value

        def _release(self):
            self.live -= 1

    planner = TrackingPlanner()
    source = {f"parameter_{i}": torch.ones(16, dtype=torch.float32) for i in range(20)}
    dcp.save(source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128), planner=planner, no_dist=True)
    # Two 64-byte tensors fit in one file. Unreachable cycles must be collected
    # at file boundaries, and peak must not grow with the total number of files.
    assert planner.peak <= 2
    assert planner.peak < len(source)


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "pread"), reason="Requires POSIX file inspection")
def test_cache_advice_is_read_only_and_preserves_dcp_values(tmp_path, monkeypatch):
    calls = []

    def advise(fd, offset, length, advice):
        import fcntl

        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert os.fstat(fd).st_size > 0
        assert os.pread(fd, 1, 0)
        calls.append((offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", getattr(os, "POSIX_FADV_DONTNEED", 4), raising=False)
    source = {f"parameter_{i}": torch.arange(16, dtype=torch.float32) + i / 100 for i in range(8)}
    dcp.save(
        source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128, release_page_cache=True), no_dist=True
    )
    assert len(calls) == len(list(tmp_path.glob("*.distcp"))) >= 4
    assert all(row == (0, 0, os.POSIX_FADV_DONTNEED) for row in calls)
    restored = {name: torch.empty_like(value) for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    assert all(torch.equal(restored[name], value) for name, value in source.items())


@pytest.mark.skipif(
    not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"), reason="Requires POSIX cache advice"
)
def test_real_page_cache_hint_preserves_fp32_master_and_moments(tmp_path):
    source = {
        "master": torch.linspace(-0.1, 0.1, 257, dtype=torch.float32),
        "exp_avg": torch.linspace(-1e-6, 1e-6, 257, dtype=torch.float32),
        "exp_avg_sq": torch.linspace(1e-12, 1e-8, 257, dtype=torch.float32),
    }
    dcp.save(
        source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=512, release_page_cache=True), no_dist=True
    )
    restored = {name: torch.empty_like(value) for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    assert all(
        torch.equal(restored[name].view(torch.int32), value.view(torch.int32)) for name, value in source.items()
    )


def test_unsupported_cache_advice_logs_once_and_preserves_checkpoint(tmp_path, monkeypatch):
    calls = []
    messages = []

    def unsupported(*args):
        calls.append(args)
        raise OSError(errno.EOPNOTSUPP, "not supported")

    monkeypatch.setattr(os, "posix_fadvise", unsupported, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", getattr(os, "POSIX_FADV_DONTNEED", 4), raising=False)
    monkeypatch.setattr(
        sys.modules[BoundedFileSystemWriter.__module__].logger,
        "warning",
        lambda message, *args: messages.append(message % args if args else message),
    )
    source = {f"parameter_{i}": torch.arange(16, dtype=torch.float32) for i in range(8)}
    dcp.save(
        source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128, release_page_cache=True), no_dist=True
    )
    assert len(calls) == 1
    assert len(messages) == 1 and "unsupported by this filesystem" in messages[0]
    restored = {name: torch.empty_like(value) for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    assert all(torch.equal(restored[name], value) for name, value in source.items())


@pytest.mark.parametrize("error_code", [errno.EACCES, errno.EOPNOTSUPP, errno.EMFILE, errno.ENFILE, errno.ENOMEM])
def test_cache_advice_reopen_unavailable_does_not_break_save(tmp_path, monkeypatch, error_code):
    original_open = os.open
    calls = []
    messages = []

    def restricted(path, flags, *args, **kwargs):
        if str(path).endswith(".distcp") and flags & os.O_ACCMODE == os.O_RDONLY:
            calls.append(path)
            raise OSError(error_code, "reopen denied")
        return original_open(path, flags, *args, **kwargs)

    if not hasattr(os, "posix_fadvise"):
        monkeypatch.setattr(os, "posix_fadvise", lambda *args: None, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", getattr(os, "POSIX_FADV_DONTNEED", 4), raising=False)
    monkeypatch.setattr(os, "open", restricted)
    monkeypatch.setattr(
        sys.modules[BoundedFileSystemWriter.__module__].logger,
        "warning",
        lambda message, *args: messages.append(message % args if args else message),
    )
    source = {f"parameter_{i}": torch.arange(16, dtype=torch.float32) for i in range(8)}
    dcp.save(
        source, storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128, release_page_cache=True), no_dist=True
    )
    assert len(calls) == 1
    assert len(messages) == 1 and "cannot reopen a shard" in messages[0]
    monkeypatch.setattr(os, "open", original_open)
    restored = {name: torch.empty_like(value) for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    assert all(torch.equal(restored[name], value) for name, value in source.items())


def test_cache_advice_io_error_does_not_publish_metadata(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError(errno.EIO, "injected I/O failure")

    monkeypatch.setattr(os, "posix_fadvise", fail, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", getattr(os, "POSIX_FADV_DONTNEED", 4), raising=False)
    with pytest.raises(CheckpointException, match="injected I/O failure"):
        dcp.save(
            {"x": torch.ones(16)},
            storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128, release_page_cache=True),
            no_dist=True,
        )
    assert not (tmp_path / ".metadata").exists()


def test_cache_advice_missing_shard_does_not_publish_metadata(tmp_path, monkeypatch):
    original_open = os.open

    def missing(path, flags, *args, **kwargs):
        if str(path).endswith(".distcp") and flags & os.O_ACCMODE == os.O_RDONLY:
            raise OSError(errno.ENOENT, "missing completed shard")
        return original_open(path, flags, *args, **kwargs)

    if not hasattr(os, "posix_fadvise"):
        monkeypatch.setattr(os, "posix_fadvise", lambda *args: None, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", getattr(os, "POSIX_FADV_DONTNEED", 4), raising=False)
    monkeypatch.setattr(os, "open", missing)
    with pytest.raises(CheckpointException, match="missing completed shard"):
        dcp.save(
            {"x": torch.ones(16)},
            storage_writer=BoundedFileSystemWriter(tmp_path, max_bytes=128, release_page_cache=True),
            no_dist=True,
        )
    assert not (tmp_path / ".metadata").exists()


def test_missing_cache_advice_api_logs_once_and_preserves_checkpoint(tmp_path, monkeypatch):
    messages = []
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    monkeypatch.setattr(
        sys.modules[BoundedFileSystemWriter.__module__].logger,
        "warning",
        lambda message, *args: messages.append(message % args if args else message),
    )
    source = {f"parameter_{i}": torch.arange(16, dtype=torch.float32) for i in range(8)}
    writer = BoundedFileSystemWriter(tmp_path, max_bytes=128, release_page_cache=True)
    dcp.save(source, storage_writer=writer, no_dist=True)
    assert not writer.release_page_cache
    assert len(messages) == 1 and "unavailable on this platform" in messages[0]
    restored = {name: torch.empty_like(value) for name, value in source.items()}
    dcp.load(restored, checkpoint_id=tmp_path, no_dist=True)
    assert all(torch.equal(restored[name], value) for name, value in source.items())


@pytest.mark.parametrize("mode", ["roundtrip", "fail"])
@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_two_rank_gloo_advice_roundtrip_and_failure_propagation(tmp_path, mode):
    env = child_environment(RunnerCapabilities("cpu"), (), cpu=True)
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    worker = Path(__file__).with_name("bounded_writer_gloo_worker.py")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(worker),
            str(tmp_path / mode),
            mode,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert f"BOUNDED_DCP_GLOO_PASS {mode}" in result.stdout
