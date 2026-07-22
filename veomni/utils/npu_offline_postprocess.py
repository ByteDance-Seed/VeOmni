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

"""Offline Ascend profile post-processing (copy / analyse / upload).

Designed to run outside the training critical path. Training can spawn this as a
fire-and-forget sidecar after raw `*_ascend_pt` finalization so peer ranks are
not blocked on Chrome/DB export or Merlin upload.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger("veomni.npu_offline_postprocess")


def _is_ascend_pt_dir(path: Path) -> bool:
    return path.is_dir() and path.name.endswith("_ascend_pt")


def resolve_raw_dir(raw_dir: str | Path) -> Path:
    """Return the concrete `*_ascend_pt` directory."""
    path = Path(raw_dir).expanduser().resolve()
    if _is_ascend_pt_dir(path):
        return path
    if path.is_dir():
        matches = sorted(p for p in path.iterdir() if _is_ascend_pt_dir(p))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Multiple *_ascend_pt directories under {path}; pass one explicitly: {[p.name for p in matches]}"
            )
    raise FileNotFoundError(f"No *_ascend_pt directory found at {raw_dir}")


def resolve_analyse_path(raw_dir: Path) -> str:
    """Target exactly one finalized ``*_ascend_pt`` capture."""
    return str(raw_dir)


def find_trace_view(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json",
        raw_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(raw_dir.rglob("trace_view.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"trace_view.json not found under {raw_dir} after analyse")


def gzip_trace(trace_path: Path) -> Path:
    if trace_path.suffix == ".gz":
        return trace_path
    out = Path(f"{trace_path}.gz")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(trace_path, "rb") as src, gzip.open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def copy_raw_dir(raw_dir: Path, copy_to: str) -> None:
    if copy_to.startswith("hdfs://"):
        try:
            from veomni.utils.hdfs_io import copy, exists, makedirs
        except Exception as exc:  # pragma: no cover - import path depends on install
            raise RuntimeError(f"HDFS copy requires veomni.utils.hdfs_io: {exc}") from exc
        target = f"{copy_to.rstrip('/')}/{raw_dir.name}"
        if exists(target):
            raise FileExistsError(f"Refusing to overwrite existing HDFS profile target: {target}")
        makedirs(copy_to, exist_ok=True)
        if not copy(str(raw_dir), target):
            raise RuntimeError(f"Failed to copy {raw_dir} to {target}")
        logger.info("Copied raw profile to %s", target)
        return

    source = raw_dir.resolve()
    dest = Path(copy_to).expanduser().resolve()
    target = dest / raw_dir.name
    if target == source:
        raise ValueError(f"copy_to cannot be the raw profile source parent: {dest}")
    try:
        dest.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError(f"copy_to cannot be inside the raw profile directory: {dest}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing profile target: {target} already exists")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    logger.info("Copied raw profile to %s", target)


def run_analyse(raw_dir: Path, max_process_number: Optional[int] = None) -> None:
    try:
        from torch_npu.profiler.profiler import analyse as npu_analyse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch_npu is required for --analyse") from exc

    profiler_path = resolve_analyse_path(raw_dir)
    trace_path = raw_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if trace_path.is_file():
        existing_stat = trace_path.stat()
        existing_signature = (
            existing_stat.st_ino,
            existing_stat.st_size,
            existing_stat.st_mtime_ns,
            existing_stat.st_ctime_ns,
        )
    else:
        existing_signature = None
    kwargs = {}
    if max_process_number is not None:
        kwargs["max_process_number"] = max_process_number
    logger.info("Running torch_npu.profiler.profiler.analyse(profiler_path=%s)", profiler_path)
    started = time.perf_counter()
    npu_analyse(profiler_path=profiler_path, **kwargs)
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace_view.json not found under {raw_dir} after analyse")
    trace_stat = trace_path.stat()
    if trace_stat.st_size == 0:
        raise RuntimeError(f"torch_npu analyse produced an empty trace: {trace_path}")
    trace_signature = (
        trace_stat.st_ino,
        trace_stat.st_size,
        trace_stat.st_mtime_ns,
        trace_stat.st_ctime_ns,
    )
    if trace_signature == existing_signature:
        raise RuntimeError(f"torch_npu analyse did not create or update the existing trace: {trace_path}")
    with open(trace_path, "rb") as trace_file:
        head = trace_file.read(4096).lstrip()
        trace_file.seek(max(0, trace_stat.st_size - 4096))
        tail = trace_file.read().rstrip()
    if not head.startswith(b"[") or not tail.endswith(b"]"):
        raise RuntimeError(f"torch_npu analyse produced an incomplete trace: {trace_path}")
    logger.info(
        "NPU_PROFILE_ANALYSE mode=offline duration_seconds=%.6f raw_dir=%s trace=%s",
        time.perf_counter() - started,
        raw_dir,
        trace_path,
    )


def run_upload_cmd(upload_cmd: str | list[str], trace_file: Path) -> None:
    if isinstance(upload_cmd, list):
        logger.info("Uploading with argv: %s", shlex.join(upload_cmd))
        subprocess.run(upload_cmd, check=True)
        return

    quoted_trace = shlex.quote(str(trace_file))
    if "{trace}" in upload_cmd:
        command = upload_cmd.replace("{trace}", quoted_trace)
    else:
        command = f"{upload_cmd} {quoted_trace}"
    logger.info("Uploading with command: %s", command)
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


def build_merlin_upload_cmd(trace_file: Path, name: Optional[str] = None) -> list[str]:
    payload = {
        "file_path": str(trace_file),
        "asset_type": "perfetto",
        "compress": False if str(trace_file).endswith(".gz") else True,
    }
    if name:
        payload["name"] = name
    # The JobRun Profiling tab lists assets by its selected Arnold trial.
    job_id = os.getenv("RH2_JOB_RUN_ID") or os.getenv("MERLIN_JOB_ID")
    trial_id = os.getenv("ARNOLD_TRIAL_ID")
    if trial_id:
        payload["trial_id"] = trial_id
    elif job_id:
        payload["job_id"] = job_id
    return ["merlin-cli", "profiling", "upload", "--json", json.dumps(payload, ensure_ascii=False)]


def upload_merlin_profile(trace_file: Path, name: Optional[str] = None) -> None:
    """Upload through Merlin's file-based CLI and fail safely when unavailable."""
    if shutil.which("merlin-cli"):
        run_upload_cmd(build_merlin_upload_cmd(trace_file, name=name), trace_file)
        return

    raise RuntimeError(
        "merlin-cli is unavailable. The bytedmerlin ProfilingAsset SDK fallback is intentionally unsupported "
        "because its JSON/base64 upload is unsafe for large traces. No upload was attempted; the raw profile "
        f"and packed trace remain on disk (trace={trace_file})."
    )


def postprocess(
    raw_dir: str | Path,
    *,
    copy_to: Optional[str] = None,
    analyse: bool = False,
    upload_cmd: Optional[str] = None,
    merlin_upload: bool = False,
    upload_name: Optional[str] = None,
    max_process_number: Optional[int] = None,
) -> Optional[Path]:
    resolved = resolve_raw_dir(raw_dir)
    logger.info("Post-processing Ascend raw profile at %s", resolved)

    if copy_to:
        copy_raw_dir(resolved, copy_to)

    if analyse:
        run_analyse(resolved, max_process_number=max_process_number)

    if not (upload_cmd or merlin_upload):
        return None

    if not analyse:
        # Allow upload of an already-parsed capture.
        logger.info("Upload requested without --analyse; expecting an existing trace_view.json")

    trace_path = find_trace_view(resolved)
    packed = gzip_trace(trace_path)
    if merlin_upload:
        upload_merlin_profile(packed, name=upload_name)
    else:
        assert upload_cmd is not None
        run_upload_cmd(upload_cmd, packed)
    return packed


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy / analyse / upload an Ascend offline raw profile outside training."
    )
    parser.add_argument("--raw-dir", required=True, help="Path to *_ascend_pt or its parent directory")
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Durable destination for the raw directory (local path or hdfs://...)",
    )
    parser.add_argument(
        "--analyse",
        action="store_true",
        help="Run torch_npu.profiler.profiler.analyse to produce Chrome/DB outputs",
    )
    parser.add_argument(
        "--upload-cmd",
        default=None,
        help="Shell command to upload the parsed trace. Use {trace} or append the path.",
    )
    parser.add_argument(
        "--merlin-upload",
        action="store_true",
        help="Upload the parsed Chrome trace with the file-based Merlin CLI.",
    )
    parser.add_argument("--upload-name", default=None, help="Optional Merlin asset display name")
    parser.add_argument("--max-process-number", type=int, default=None, help="torch_npu analyse parallelism")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)
    if not (args.copy_to or args.analyse or args.upload_cmd or args.merlin_upload):
        logger.error("Nothing to do: pass --copy-to and/or --analyse and/or an upload option")
        return 2
    postprocess(
        args.raw_dir,
        copy_to=args.copy_to,
        analyse=args.analyse,
        upload_cmd=args.upload_cmd,
        merlin_upload=args.merlin_upload,
        upload_name=args.upload_name,
        max_process_number=args.max_process_number,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
