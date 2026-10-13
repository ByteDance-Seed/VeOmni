import subprocess
from types import SimpleNamespace

import pytest

from veomni.utils import hdfs_io


HDFS_DIR = "hdfs://cluster/data/train"

LS_OUTPUT = """Found 3 items
-rw-r--r--   3 user group       1024 2026-10-01 12:00 hdfs://cluster/data/train/part-0.parquet
-rw-r--r--   3 user group       2048 2026-10-01 12:00 hdfs://cluster/data/train/my part-1.parquet
drwxr-xr-x   - user group          0 2026-10-01 12:00 hdfs://cluster/data/train/sub/
"""


@pytest.fixture(autouse=True)
def hdfs_binary(monkeypatch):
    monkeypatch.setattr(hdfs_io, "_HDFS_BIN_PATH", "/usr/bin/hdfs")


@pytest.mark.parametrize(("returncode", "expected"), [(0, True), (1, False)])
def test_isdir_follows_the_hdfs_test_exit_code(monkeypatch, returncode, expected):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout="", stderr=""),
    )

    assert hdfs_io.isdir(HDFS_DIR) is expected


def test_listdir_returns_the_entry_names(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=LS_OUTPUT, stderr=""),
    )

    assert hdfs_io.listdir(HDFS_DIR) == ["part-0.parquet", "my part-1.parquet", "sub"]


def test_listdir_raises_when_the_listing_fails(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="No such file or directory"),
    )

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        hdfs_io.listdir(HDFS_DIR)


TRICKY_PATH = "hdfs://cluster/data/my dir; touch /tmp/pwned $(id) `id`"


def _capture_run(monkeypatch, stdout=""):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(hdfs_io, "_HDFS_BIN_PATH", "/usr/bin/hdfs")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_listdir_passes_the_path_as_one_argument_without_a_shell(monkeypatch):
    calls = _capture_run(monkeypatch)

    hdfs_io.listdir(TRICKY_PATH)

    args, kwargs = calls[0]
    assert args == ["/usr/bin/hdfs", "dfs", "-ls", TRICKY_PATH]
    assert not kwargs.get("shell")


def test_isdir_passes_the_path_as_one_argument_without_a_shell(monkeypatch):
    calls = _capture_run(monkeypatch)

    hdfs_io.isdir(TRICKY_PATH)

    args, kwargs = calls[0]
    assert args == ["/usr/bin/hdfs", "dfs", "-test", "-d", TRICKY_PATH]
    assert not kwargs.get("shell")


def test_listdir_raises_when_the_hdfs_executable_is_missing(monkeypatch):
    monkeypatch.setattr(hdfs_io, "_HDFS_BIN_PATH", None)

    with pytest.raises(FileNotFoundError, match="hdfs executable"):
        hdfs_io.listdir(HDFS_DIR)
