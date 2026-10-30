"""HF checkpoint page-cache advice must not change loaded values."""

import ctypes
import errno
import json
import mmap
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec, RunnerCapabilities
from scripts.ci.accelerator.runtime.devices import child_environment
from tests.tools.launch_utils import run_bounded_process_group
from veomni.models.checkpoint import weights


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_hf_read_cache_disabled_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VEOMNI_HF_DROP_READ_CACHE", raising=False)
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"original")
    monkeypatch.setattr(weights.os, "open", lambda *args: pytest.fail("disabled advice opened a file"))
    assert weights._release_hf_read_cache([source]) == {
        "enabled": False,
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
    }


def test_hf_read_cache_load_preserves_model_and_source(monkeypatch, tmp_path):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    original = torch.nn.Linear(4, 4)
    weights.save_model_weights(str(tmp_path), original.state_dict(), global_rank=0, save_dtype="float32")
    source = tmp_path / "model.safetensors"
    source_bytes = source.read_bytes()
    calls = []
    monkeypatch.setattr(
        weights.os, "posix_fadvise", lambda fd, start, length, advice: calls.append((start, length, advice))
    )
    restored = torch.nn.Linear(4, 4)
    restored.config = SimpleNamespace()
    weights.load_model_weights(restored, str(tmp_path), init_device="cpu")
    assert calls == [(0, 0, os.POSIX_FADV_DONTNEED)]
    assert source.read_bytes() == source_bytes
    assert torch.equal(restored.weight, original.weight)
    assert torch.equal(restored.bias, original.bias)


def test_hf_read_cache_advice_failure_does_not_modify_source(monkeypatch, tmp_path):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"original")

    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "unsupported")

    monkeypatch.setattr(weights.os, "posix_fadvise", unsupported)
    assert weights._release_hf_read_cache([source, source]) == {
        "enabled": True,
        "attempted": 1,
        "succeeded": 0,
        "failed": 1,
    }
    assert source.read_bytes() == b"original"


def test_hf_read_cache_invalid_option_disables_advice(monkeypatch):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "unknown")
    assert weights._release_hf_read_cache([]) == {"enabled": False, "attempted": 0, "succeeded": 0, "failed": 0}


def test_hf_read_cache_close_failure_is_best_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"original")
    close = os.close

    def close_then_report_error(fd):
        close(fd)
        raise OSError(errno.EIO, "simulated close error")

    monkeypatch.setattr(weights.os, "posix_fadvise", lambda *args: None)
    monkeypatch.setattr(weights.os, "close", close_then_report_error)
    result = weights._release_hf_read_cache([source])
    assert result["attempted"] == 1 and result["succeeded"] == 0 and result["failed"] == 1
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("mode", ["standard", "ep"])
def test_advice_uses_actual_resolved_revision_once(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    pinned = tmp_path / "pinned.safetensors"
    default = tmp_path / "default.safetensors"
    pinned.write_bytes(b"pinned")
    default.write_bytes(b"default")
    resolver_calls, advised = [], []

    @weights._capture_hf_checkpoint_paths(mode)
    def resolver(path, *, revision):
        resolver_calls.append(revision)
        selected = pinned if revision == "pinned" else default
        if mode == "ep":
            return {}, {selected.name: str(selected)}
        return [weights.StateDictIterator(str(selected))]

    @weights._advise_after_hf_load(mode)
    def loader(model, path, *, revision):
        resolver(path, revision=revision)
        return "loaded"

    monkeypatch.setattr(weights, "_release_hf_read_cache", lambda paths: advised.extend(paths))
    assert loader(None, str(tmp_path), revision="pinned") == "loaded"
    assert resolver_calls == ["pinned"]
    assert advised == [str(pinned)]


@pytest.mark.parametrize("rank,expected_calls", [(0, 1), (1, 0)])
def test_rank0_broadcast_releases_only_on_the_reader(monkeypatch, tmp_path, rank, expected_calls):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    source_model = torch.nn.Linear(4, 4)
    weights.save_model_weights(str(tmp_path), source_model.state_dict(), global_rank=0, save_dtype="float32")
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(weights.os, "posix_fadvise", lambda *args: calls.append(args))

    def read_on_rank_zero(model, path):
        if rank == 0:
            weights._resolve_state_dict_iterators(path)
        return "loaded"

    loader = weights._advise_after_hf_load("broadcast")(read_on_rank_zero)
    assert loader(source_model, str(tmp_path)) == "loaded"
    assert len(calls) == expected_calls


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX file advice")
@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("invalid_rank", [None, 0, 1])
def test_rank0_broadcast_gloo_preserves_weights_and_advises_once(monkeypatch, tmp_path, invalid_rank):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    if invalid_rank is not None:
        monkeypatch.setenv("VEOMNI_TEST_INVALID_CACHE_RANK", str(invalid_rank))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original = torch.nn.Linear(4, 4)
    with torch.no_grad():
        original.weight.fill_(1.5)
        original.bias.fill_(0.25)
    weights.save_model_weights(str(checkpoint), original.state_dict(), global_rank=0, save_dtype="float32")
    source = checkpoint / "model.safetensors"
    source_bytes = source.read_bytes()
    output = tmp_path / "result.json"
    worker = Path(__file__).with_name("hf_read_cache_gloo_worker.py")
    env = child_environment(RunnerCapabilities("cpu"), (), cpu=True)
    run_bounded_process_group(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
            str(checkpoint),
            str(output),
        ],
        timeout=180,
        env=env,
    )
    rows = json.loads(output.read_text())
    assert rows == [
        {"rank": 0, "equal": True, "paths": [] if invalid_rank == 0 else [str(source.resolve())]},
        {"rank": 1, "equal": True, "paths": []},
    ]
    assert source.read_bytes() == source_bytes


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux mincore and /proc/self/maps")
def test_hf_advice_runs_after_source_mmap_is_released(monkeypatch, tmp_path):
    monkeypatch.setenv("VEOMNI_HF_DROP_READ_CACHE", "1")
    original = torch.nn.Linear(2048, 2048, bias=True)
    weights.save_model_weights(str(tmp_path), original.state_dict(), global_rank=0, save_dtype="float32")
    source = tmp_path / "model.safetensors"
    fd = os.open(source, os.O_RDONLY)
    os.fsync(fd)
    size = os.fstat(fd).st_size
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    address = libc.mmap(None, size, 0, 1, fd, 0)
    if address == ctypes.c_void_p(-1).value:
        os.close(fd)
        raise OSError(ctypes.get_errno(), "mmap")

    def resident_pages():
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        flags = (ctypes.c_ubyte * pages)()
        if libc.mincore(address, size, flags):
            raise OSError(ctypes.get_errno(), "mincore")
        return sum(bool(value & 1) for value in flags)

    advice, observed = weights._release_hf_read_cache, {}

    def measure(paths):
        # The PROT_NONE mapping above accounts for one mapping; a second
        # mapping of this file would reveal a still-live safetensors tensor.
        with open("/proc/self/maps") as process_maps:
            observed["maps"] = process_maps.read().count(str(source))
        observed["before"] = resident_pages()
        observed["result"] = advice(paths)
        observed["after"] = resident_pages()

    monkeypatch.setattr(weights, "_release_hf_read_cache", measure)
    restored = torch.nn.Linear(2048, 2048, bias=True)
    restored.config = SimpleNamespace()
    try:
        weights.load_model_weights(restored, str(tmp_path), init_device="cpu")
        assert observed["maps"] == 1
        assert observed["result"]["failed"] == 0
        assert observed["after"] <= observed["before"]
        if os.getenv("VEOMNI_REQUIRE_FADVISE_RESIDENCY_TEST") == "1":
            assert observed["before"] > 0 and observed["after"] < observed["before"] // 2
        assert torch.equal(restored.weight, original.weight)
        assert torch.equal(restored.bias, original.bias)
    finally:
        libc.munmap(address, size)
        os.close(fd)
