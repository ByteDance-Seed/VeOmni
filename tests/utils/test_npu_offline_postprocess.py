import json
import shlex

import pytest

from veomni.utils import hdfs_io
from veomni.utils import npu_offline_postprocess as post


def test_resolve_raw_dir_accepts_ascend_pt(tmp_path):
    raw = tmp_path / "localhost_1_2_ascend_pt"
    raw.mkdir()
    assert post.resolve_raw_dir(raw) == raw.resolve()


def test_resolve_raw_dir_finds_unique_child(tmp_path):
    raw = tmp_path / "localhost_1_2_ascend_pt"
    raw.mkdir()
    assert post.resolve_raw_dir(tmp_path) == raw.resolve()


def test_resolve_raw_dir_rejects_ambiguous_parent(tmp_path):
    (tmp_path / "rank0_1_ascend_pt").mkdir()
    (tmp_path / "rank0_2_ascend_pt").mkdir()

    with pytest.raises(FileNotFoundError, match="Multiple"):
        post.resolve_raw_dir(tmp_path)


def test_resolve_analyse_path_targets_only_requested_capture(tmp_path):
    raw = tmp_path / "localhost_1_2_ascend_pt"
    sibling = tmp_path / "localhost_3_4_ascend_pt"
    raw.mkdir()
    sibling.mkdir()

    assert post.resolve_analyse_path(raw) == str(raw)


def test_copy_raw_dir_refuses_source_parent_and_existing_target(tmp_path):
    raw = tmp_path / "rank0_ascend_pt"
    raw.mkdir()
    marker = raw / "raw.bin"
    marker.write_bytes(b"profile")

    with pytest.raises(ValueError, match="source parent"):
        post.copy_raw_dir(raw, str(tmp_path))
    assert marker.read_bytes() == b"profile"

    destination = tmp_path / "durable"
    target = destination / raw.name
    target.mkdir(parents=True)
    keep = target / "keep.txt"
    keep.write_text("do-not-delete", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        post.copy_raw_dir(raw, str(destination))
    assert keep.read_text(encoding="utf-8") == "do-not-delete"


def test_copy_raw_dir_refuses_existing_hdfs_target(tmp_path, monkeypatch):
    raw = tmp_path / "rank0_ascend_pt"
    raw.mkdir()
    copies = []
    target = "hdfs://haruna/profile/rank0_ascend_pt"
    monkeypatch.setattr(hdfs_io, "exists", lambda path: path == target)
    monkeypatch.setattr(hdfs_io, "copy", lambda *args: copies.append(args))

    with pytest.raises(FileExistsError, match="HDFS profile target"):
        post.copy_raw_dir(raw, "hdfs://haruna/profile")

    assert copies == []


def test_gzip_and_merlin_upload_cmd(tmp_path):
    raw = tmp_path / "rank0_ascend_pt"
    out_dir = raw / "ASCEND_PROFILER_OUTPUT"
    out_dir.mkdir(parents=True)
    trace = out_dir / "trace_view.json"
    trace.write_text('{"traceEvents":[]}', encoding="utf-8")

    packed = post.gzip_trace(trace)
    assert packed.is_file()
    assert packed.name.endswith(".gz")

    cmd = post.build_merlin_upload_cmd(packed, name="npu-trace")
    assert cmd[:4] == ["merlin-cli", "profiling", "upload", "--json"]
    payload = json.loads(cmd[4])
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


def test_custom_upload_preserves_json_braces_and_quotes_trace_path(tmp_path, monkeypatch):
    trace = tmp_path / "trace with spaces.json.gz"
    trace.write_bytes(b"trace")
    calls = []
    monkeypatch.setattr(post.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    command = 'upload --json {"asset_type":"perfetto"} --file {trace}'
    post.run_upload_cmd(command, trace)

    assert calls == [
        (
            (f'upload --json {{"asset_type":"perfetto"}} --file {shlex.quote(str(trace))}',),
            {"shell": True, "check": True, "executable": "/bin/bash"},
        )
    ]


def test_merlin_upload_uses_argv_without_shell(tmp_path, monkeypatch):
    raw = tmp_path / "rank0_ascend_pt"
    out_dir = raw / "ASCEND_PROFILER_OUTPUT"
    out_dir.mkdir(parents=True)
    (out_dir / "trace_view.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(post.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    packed = post.postprocess(raw, merlin_upload=True, upload_name="npu trace")

    assert packed is not None
    argv = calls[0][0][0]
    assert argv[:4] == ["merlin-cli", "profiling", "upload", "--json"]
    assert json.loads(argv[4])["name"] == "npu trace"
    assert calls[0][1] == {"check": True}


def test_postprocess_requires_action(tmp_path):
    raw = tmp_path / "rank0_ascend_pt"
    raw.mkdir()
    assert post.main(["--raw-dir", str(raw)]) == 2
