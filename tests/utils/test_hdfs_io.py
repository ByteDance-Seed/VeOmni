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


@pytest.mark.parametrize(("returncode", "expected"), [(0, True), (1, False)])
def test_isdir_follows_the_hdfs_test_exit_code(monkeypatch, returncode, expected):
    monkeypatch.setattr(hdfs_io, "_run_cmd", lambda cmd, timeout=None: returncode)

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
