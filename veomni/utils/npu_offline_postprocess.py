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
import shutil
import subprocess
import sys
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
    """torch_npu.profiler.analyse expects the parent of `*_ascend_pt`."""
    return str(raw_dir.parent)


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
    with open(trace_path, "rb") as src, gzip.open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def copy_raw_dir(raw_dir: Path, copy_to: str) -> None:
    if copy_to.startswith("hdfs://"):
        try:
            from veomni.utils.hdfs_io import copy, makedirs
        except Exception as exc:  # pragma: no cover - import path depends on install
            raise RuntimeError(f"HDFS copy requires veomni.utils.hdfs_io: {exc}") from exc
        makedirs(copy_to, exist_ok=True)
        if not copy(str(raw_dir), copy_to):
            raise RuntimeError(f"Failed to copy {raw_dir} to {copy_to}")
        logger.info("Copied raw profile to %s", copy_to)
        return

    dest = Path(copy_to).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / raw_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(raw_dir, target)
    logger.info("Copied raw profile to %s", target)


def run_analyse(raw_dir: Path, max_process_number: Optional[int] = None) -> None:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch_npu is required for --analyse") from exc

    profiler_path = resolve_analyse_path(raw_dir)
    kwargs = {}
    if max_process_number is not None:
        kwargs["max_process_number"] = max_process_number
    logger.info("Running torch_npu.profiler.analyse(profiler_path=%s)", profiler_path)
    torch_npu.profiler.analyse(profiler_path=profiler_path, **kwargs)


def run_upload_cmd(upload_cmd: str, trace_file: Path) -> None:
    if "{trace}" in upload_cmd:
        command = upload_cmd.format(trace=str(trace_file))
    else:
        command = f"{upload_cmd} {trace_file}"
    logger.info("Uploading with command: %s", command)
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


def build_merlin_upload_cmd(trace_file: Path, name: Optional[str] = None) -> str:
    payload = {
        "file_path": str(trace_file),
        "asset_type": "perfetto",
        "compress": False if str(trace_file).endswith(".gz") else True,
    }
    if name:
        payload["name"] = name
    # job_id / trial_id are inferred by merlin-cli from RH2_JOB_RUN_ID / ARNOLD_TRIAL_ID.
    return "merlin-cli profiling upload --json " + json.dumps(payload, ensure_ascii=False)


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
    cmd = upload_cmd
    if merlin_upload:
        cmd = build_merlin_upload_cmd(packed, name=upload_name)
    assert cmd is not None
    run_upload_cmd(cmd, packed)
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
        help="Run torch_npu.profiler.analyse to produce Chrome/DB outputs",
    )
    parser.add_argument(
        "--upload-cmd",
        default=None,
        help="Shell command to upload the parsed trace. Use {trace} or append the path.",
    )
    parser.add_argument(
        "--merlin-upload",
        action="store_true",
        help="Upload the parsed Chrome trace with merlin-cli profiling upload (Perfetto).",
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
