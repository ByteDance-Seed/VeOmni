"""Storage identity, persistence, and cleanup for FP32 optimizer scratch."""

import gc
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

spec = importlib.util.spec_from_file_location(
    "file_backed_state_test", Path(__file__).parents[2] / "veomni/optim/file_backed_state.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_filesystem_type = module._filesystem_type


@pytest.fixture(autouse=True)
def use_test_filesystem_classification(monkeypatch):
    # Unit arithmetic/lifetime tests can run on overlay CI. The explicit
    # rejection tests below exercise the real classifier separately.
    monkeypatch.setattr(module, "_filesystem_type", lambda path: "ext4")


def test_file_backed_state_preserves_bits_and_alias_lifetime(tmp_path):
    store = module.FileBackedFP32State(tmp_path, minimum_free_bytes=0)
    tensor = store.zeros((12, 8))
    assert torch.equal(tensor, torch.zeros_like(tensor))
    tensor.view(torch.int32).copy_(torch.arange(tensor.numel(), dtype=torch.int32).view(tensor.shape) + 1050000000)
    expected = tensor.view(torch.int32).clone()
    alias = tensor.detach()[2:]
    assert store.owns(alias)
    fd = store._entries[tensor.untyped_storage()._cdata][1].fileno()
    assert os.fstat(fd).st_blocks * 512 >= tensor.numel() * tensor.element_size()
    assert {path.name for path in tmp_path.iterdir()} == {".veomni-fp32-reserve.lock"}
    assert store.evict(alias)
    torch.testing.assert_close(tensor.view(torch.int32), expected, rtol=0, atol=0)
    child = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import hashlib,os,sys; print(hashlib.sha256(os.fdopen(int(sys.argv[1]),'rb').read()).hexdigest())",
            str(fd),
        ],
        text=True,
        pass_fds=(fd,),
    ).strip()
    import hashlib

    assert child == hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    del tensor
    gc.collect()
    os.fstat(fd)
    del alias
    gc.collect()
    with pytest.raises(OSError):
        os.fstat(fd)
    later = store.zeros((2,))
    assert torch.equal(later, torch.zeros_like(later))


def test_file_backed_state_rejects_insufficient_space_without_files(tmp_path):
    store = module.FileBackedFP32State(tmp_path, minimum_free_bytes=1 << 100)
    with pytest.raises(RuntimeError, match="Insufficient"):
        store.zeros((16,))
    assert {path.name for path in tmp_path.iterdir()} == {".veomni-fp32-reserve.lock"}


def test_file_backed_state_separate_storage_and_foreign_noop(tmp_path):
    store = module.FileBackedFP32State(tmp_path, minimum_free_bytes=0)
    a, b = store.zeros((4, 5)), store.zeros((2, 7))
    assert a.untyped_storage()._cdata != b.untyped_storage()._cdata
    assert a.untyped_storage().nbytes() == a.numel() * a.element_size()
    assert not store.evict(torch.zeros(3))
    assert store.evict(a)


def test_range_eviction_preserves_unaligned_chunks_and_other_pages(tmp_path):
    store = module.FileBackedFP32State(tmp_path, minimum_free_bytes=0)
    tensor = store.zeros((10000,))
    tensor.copy_(torch.arange(tensor.numel(), dtype=torch.float32))
    reference = tensor.clone()
    assert store.evict_range(tensor, 11, 1234)
    tensor[8000:8010].add_(0.25)
    reference[8000:8010].add_(0.25)
    assert store.evict_range(tensor, 8000, 10)
    assert torch.equal(tensor.view(torch.int32), reference.view(torch.int32))
    with pytest.raises(ValueError, match="range"):
        store.evict_range(tensor, 9999, 2)


def test_file_backed_state_rejects_memory_backed_scratch(monkeypatch):
    monkeypatch.setattr(module, "_filesystem_type", real_filesystem_type)
    if not Path("/dev/shm").is_dir():
        return
    if module._filesystem_type("/dev/shm") == "tmpfs":
        with pytest.raises(ValueError, match="local disk"):
            module.FileBackedFP32State("/dev/shm", minimum_free_bytes=0)


def test_file_backed_state_rejects_unresolved_overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_filesystem_type", lambda path: "overlay")
    with pytest.raises(ValueError, match="local disk"):
        module.FileBackedFP32State(tmp_path, minimum_free_bytes=0)


def test_empty_local_shard_needs_no_file(tmp_path):
    store = module.FileBackedFP32State(tmp_path, minimum_free_bytes=0)
    tensor = store.zeros((0, 128))
    assert tensor.shape == (0, 128) and tensor.dtype == torch.float32
    assert store.owns(tensor) and store.evict(tensor)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="negative"):
        store.zeros((-1, 128))


def test_abrupt_exit_leaves_no_scratch_file(tmp_path):
    source = str(Path(__file__).parents[2] / "veomni/optim/file_backed_state.py")
    script = (
        "import importlib.util,os,sys; "
        "spec=importlib.util.spec_from_file_location('mapped',sys.argv[1]); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "m._filesystem_type=lambda path: 'ext4'; "
        "store=m.FileBackedFP32State(sys.argv[2],minimum_free_bytes=0); "
        "tensor=store.zeros((1024*1024,)); tensor.fill_(1); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script, source, str(tmp_path)], check=True)
    assert {path.name for path in tmp_path.iterdir()} == {".veomni-fp32-reserve.lock"}


def test_concurrent_reservations_do_not_consume_the_shared_reserve(tmp_path):
    source = str(Path(__file__).parents[2] / "veomni/optim/file_backed_state.py")
    counter = tmp_path / "simulated-free-bytes"
    counter.write_text(str(12288))
    script = """
import importlib.util, os, pathlib, sys, time, types
spec = importlib.util.spec_from_file_location('mapped', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._filesystem_type = lambda path: 'ext4'
counter = pathlib.Path(sys.argv[2]) / 'simulated-free-bytes'
real_fallocate = os.posix_fallocate
def fake_statvfs(path):
    return types.SimpleNamespace(f_bavail=int(counter.read_text()), f_frsize=1)
def fake_fallocate(fd, offset, size):
    remaining = int(counter.read_text())
    time.sleep(0.2)
    counter.write_text(str(remaining - size))
    real_fallocate(fd, offset, size)
module.os.statvfs = fake_statvfs
module.os.posix_fallocate = fake_fallocate
store = module.FileBackedFP32State(sys.argv[2], minimum_free_bytes=4096)
try:
    tensor = store.zeros((2048,))
    print('allocated', flush=True)
except RuntimeError as error:
    if 'Insufficient' not in str(error): raise
    print('reserve-protected', flush=True)
"""
    processes = [
        subprocess.Popen([sys.executable, "-c", script, source, str(tmp_path)], stdout=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    results = [process.communicate(timeout=30)[0].strip() for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert sorted(results) == ["allocated", "reserve-protected"]
    assert int(counter.read_text()) == 4096
