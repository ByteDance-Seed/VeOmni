import json
from pathlib import Path

import pytest

from veomni.utils import npu_offline_postprocess as post


def test_resolve_raw_dir_accepts_ascend_pt(tmp_path):
    raw = tmp_path / "localhost_1_2_ascend_pt"
    raw.mkdir()
    assert post.resolve_raw_dir(raw) == raw.resolve()


def test_resolve_raw_dir_finds_unique_child(tmp_path):
    raw = tmp_path / "localhost_1_2_ascend_pt"
    raw.mkdir()
    assert post.resolve_raw_dir(tmp_path) == raw.resolve()


def test_gzip_and_merlin_upload_cmd(tmp_path, monkeypatch):
    raw = tmp_path / "rank0_ascend_pt"
    out_dir = raw / "ASCEND_PROFILER_OUTPUT"
    out_dir.mkdir(parents=True)
    trace = out_dir / "trace_view.json"
    trace.write_text('{"traceEvents":[]}', encoding="utf-8")

    packed = post.gzip_trace(trace)
    assert packed.is_file()
    assert packed.name.endswith(".gz")

    cmd = post.build_merlin_upload_cmd(packed, name="npu-trace")
    assert cmd.startswith("merlin-cli profiling upload --json ")
    payload = json.loads(cmd.split(" --json ", 1)[1])
    assert payload["file_path"] == str(packed)
    assert payload["asset_type"] == "perfetto"
    assert payload["name"] == "npu-trace"


def test_postprocess_upload_without_analyse_uses_existing_trace(tmp_path, monkeypatch):
    raw = tmp_path / "rank0_ascend_pt"
    out_dir = raw / "ASCEND_PROFILER_OUTPUT"
    out_dir.mkdir(parents=True)
    trace = out_dir / "trace_view.json"
    trace.write_text("{}", encoding="utf-8")
    uploads = []

    monkeypatch.setattr(post, "run_upload_cmd", lambda cmd, path: uploads.append((cmd, path)))

    packed = post.postprocess(raw, analyse=False, upload_cmd="upload-trace {trace}")
    assert packed is not None
    assert packed.name.endswith(".gz")
    assert uploads == [("upload-trace {trace}", packed)]


def test_postprocess_requires_action(tmp_path):
    raw = tmp_path / "rank0_ascend_pt"
    raw.mkdir()
    assert post.main(["--raw-dir", str(raw)]) == 2
