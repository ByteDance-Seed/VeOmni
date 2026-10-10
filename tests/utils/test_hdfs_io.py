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

"""Tests for the built-in HDFS fallback in veomni.utils.hdfs_io (no HDFS service needed)."""

from unittest.mock import patch

import pytest

from veomni.utils import hdfs_io


HDFS_DIR = "hdfs://example/data"


@pytest.fixture(autouse=True)
def hdfs_bin():
    with patch.object(hdfs_io, "_HDFS_BIN_PATH", "/mock/hdfs"):
        yield


@pytest.mark.parametrize("returncode, expected", [(0, True), (256, False)])
def test_isdir_reflects_test_command_result(returncode, expected):
    with patch.object(hdfs_io, "_run_cmd", return_value=returncode) as run_cmd:
        assert hdfs_io.isdir(HDFS_DIR) is expected
    run_cmd.assert_called_once_with(f"/mock/hdfs dfs -test -d {HDFS_DIR}")


def test_listdir_returns_entry_names():
    output = f"{HDFS_DIR}/part-0.parquet\n{HDFS_DIR}/part-1.parquet\n{HDFS_DIR}/sub/\n"
    with patch.object(hdfs_io, "_run_cmd_output", return_value=(0, output)) as run_cmd:
        entries = hdfs_io.listdir(HDFS_DIR)
    run_cmd.assert_called_once_with(f"/mock/hdfs dfs -ls -C {HDFS_DIR}")
    assert entries == ["part-0.parquet", "part-1.parquet", "sub"]
    assert sorted(entries) == ["part-0.parquet", "part-1.parquet", "sub"]


def test_listdir_empty_directory():
    with patch.object(hdfs_io, "_run_cmd_output", return_value=(0, "")):
        assert hdfs_io.listdir(HDFS_DIR) == []


def test_listdir_failure_raises():
    with patch.object(hdfs_io, "_run_cmd_output", return_value=(1, "")):
        with pytest.raises(FileNotFoundError, match="Failed to list"):
            hdfs_io.listdir(HDFS_DIR)


def test_local_paths_unchanged(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}")
    assert hdfs_io.isdir(str(tmp_path))
    assert hdfs_io.listdir(str(tmp_path)) == ["a.jsonl"]
