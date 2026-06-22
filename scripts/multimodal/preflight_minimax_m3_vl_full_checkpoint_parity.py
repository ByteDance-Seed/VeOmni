#!/usr/bin/env python3
"""Preflight MiniMax M3 VL full-checkpoint forward parity on target machines."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_MIN_PAYLOAD_BYTES = 800_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--candidate-device", default="npu")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--require-free-disk-gb", type=float, default=0.0)
    parser.add_argument("--require-free-hbm-mb", type=int, default=0)
    parser.add_argument("--npu-smi-cmd", default=os.environ.get("MINIMAX_NPU_SMI_CMD", ""))
    parser.add_argument("--expected-shards", type=int, default=59)
    parser.add_argument("--expected-min-weight-map-keys", type=int, default=20_000)
    parser.add_argument("--expected-min-payload-bytes", type=int, default=DEFAULT_MIN_PAYLOAD_BYTES)
    return parser.parse_args()


def add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def version_at_least(version: str | None, minimum: tuple[int, int, int]) -> bool:
    if not version:
        return False
    parts = tuple(int(part) for part in re.findall(r"\d+", version)[:3])
    return parts >= minimum if len(parts) == 3 else False


def device_kind(device: str) -> str:
    return device.split(":", 1)[0].lower()


def find_existing_parent(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def disk_report(path: Path) -> dict[str, Any]:
    existing = find_existing_parent(path)
    usage = shutil.disk_usage(existing)
    return {
        "path": str(path),
        "checked_path": str(existing),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gb": usage.free / (1024**3),
    }


def run_command(command: str) -> dict[str, Any] | None:
    if not command:
        return None
    try:
        completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return {"command": command, "returncode": None, "error": repr(exc), "output_excerpt": ""}
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_excerpt": output[:12000],
        "truncated": len(output) > 12000,
    }


def default_npu_smi_command() -> str:
    for candidate in ("npu-smi", "/usr/local/sbin/npu-smi", "/usr/local/bin/npu-smi"):
        if shutil.which(candidate) or os.path.exists(candidate):
            return f"{candidate} info"
    return ""


def parse_npu_hbm(output: str) -> list[dict[str, int]]:
    devices = []
    current_device: int | None = None
    for line in output.splitlines():
        if "Process id" in line:
            break
        device_match = re.match(r"\|\s*(\d+)\s+\S+\s+\|", line)
        if device_match and "OK" in line:
            current_device = int(device_match.group(1))
            continue
        if current_device is None:
            continue
        pairs = [(int(used), int(total)) for used, total in re.findall(r"(\d+)\s*/\s*(\d+)", line)]
        hbm_pairs = [(used, total) for used, total in pairs if total >= 1024]
        if hbm_pairs:
            used, total = hbm_pairs[-1]
            devices.append({"device": current_device, "used_mb": used, "total_mb": total, "free_mb": total - used})
            current_device = None
    return devices


def checkpoint_report(checkpoint_dir: Path) -> dict[str, Any]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "index_json": str(index_path),
        "index_exists": index_path.exists(),
        "weight_map_keys": 0,
        "selected_shard_count": 0,
        "missing_shards": [],
        "payload_bytes_present": 0,
        "shards": [],
    }
    if not index_path.exists():
        return report

    index_data = json.loads(index_path.read_text())
    weight_map = index_data.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    shards = []
    missing = []
    payload_bytes_present = 0
    for shard_name in shard_names:
        shard_path = checkpoint_dir / shard_name
        exists = shard_path.exists()
        size = shard_path.stat().st_size if exists else 0
        payload_bytes_present += size
        if not exists:
            missing.append(shard_name)
        shards.append({"name": shard_name, "exists": exists, "bytes": size})

    report.update(
        {
            "weight_map_keys": len(weight_map),
            "selected_shard_count": len(shard_names),
            "missing_shards": missing,
            "payload_bytes_present": payload_bytes_present,
            "shards": shards,
        }
    )
    return report


def runtime_report(reference_device: str, candidate_device: str, npu_smi_cmd: str, required_free_hbm_mb: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "reference_device": reference_device,
        "candidate_device": candidate_device,
        "torch_version": None,
        "transformers_version": None,
        "torch_npu_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "npu_available": False,
        "npu_device_count": 0,
        "npu_smi": None,
        "ascend_env": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "MODELING_BACKEND",
            )
        },
        "import_errors": [],
    }

    needs_npu = "npu" in {device_kind(reference_device), device_kind(candidate_device)}
    if needs_npu or required_free_hbm_mb > 0:
        command = npu_smi_cmd or default_npu_smi_command()
        npu_smi = run_command(command)
        if npu_smi is not None:
            npu_smi["hbm"] = parse_npu_hbm(npu_smi.get("output_excerpt", ""))
            npu_smi["required_free_hbm_mb"] = required_free_hbm_mb
            npu_smi["devices_with_required_free_hbm"] = sum(
                1 for device in npu_smi["hbm"] if device["free_mb"] >= required_free_hbm_mb
            )
        report["npu_smi"] = npu_smi

    try:
        import torch
    except Exception as exc:
        torch = None
        report["import_errors"].append(f"torch import failed: {exc!r}")
    else:
        report["torch_version"] = torch.__version__
        if hasattr(torch, "cuda"):
            report["cuda_available"] = bool(torch.cuda.is_available())
            report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if hasattr(torch, "npu"):
            try:
                report["npu_available"] = bool(torch.npu.is_available())
                report["npu_device_count"] = int(torch.npu.device_count()) if torch.npu.is_available() else 0
            except Exception as exc:
                report["import_errors"].append(f"torch.npu probe failed: {exc!r}")

    try:
        import transformers
    except Exception as exc:
        report["import_errors"].append(f"transformers import failed: {exc!r}")
    else:
        report["transformers_version"] = transformers.__version__

    if needs_npu and torch is not None:
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            report["import_errors"].append(f"torch_npu import failed: {exc!r}")
        else:
            report["torch_npu_version"] = getattr(torch_npu, "__version__", None)

    return report


def validate_device(runtime: dict[str, Any], device: str, issues: list[str]) -> None:
    kind = device_kind(device)
    if kind == "cpu":
        return
    if kind == "cuda":
        add_issue(issues, runtime.get("cuda_available") is True, f"device {device} requested but CUDA is not available")
        add_issue(issues, runtime.get("cuda_device_count", 0) >= 1, f"device {device} requested but no CUDA devices are visible")
        return
    if kind == "npu":
        add_issue(issues, runtime.get("npu_available") is True, f"device {device} requested but NPU is not available")
        add_issue(issues, runtime.get("npu_device_count", 0) >= 1, f"device {device} requested but no NPU devices are visible")
        add_issue(issues, runtime.get("torch_npu_version") is not None, f"device {device} requested but torch_npu is missing")
        ascend_env = runtime.get("ascend_env") or {}
        add_issue(
            issues,
            ascend_env.get("ASCEND_RT_VISIBLE_DEVICES") is not None or ascend_env.get("ASCEND_VISIBLE_DEVICES") is not None,
            f"device {device} requested but visible Ascend device env is missing",
        )
        return
    issues.append(f"unsupported device kind: {device}")


def main() -> None:
    args = parse_args()
    issues: list[str] = []

    checkpoint = checkpoint_report(args.checkpoint_dir)
    add_issue(issues, args.checkpoint_dir.is_dir(), f"checkpoint dir does not exist: {args.checkpoint_dir}")
    add_issue(issues, checkpoint["index_exists"], "model.safetensors.index.json is missing")
    add_issue(
        issues,
        checkpoint["weight_map_keys"] >= args.expected_min_weight_map_keys,
        f"weight_map has fewer than {args.expected_min_weight_map_keys} keys",
    )
    add_issue(
        issues,
        checkpoint["selected_shard_count"] >= args.expected_shards,
        f"checkpoint has fewer than {args.expected_shards} safetensors shards",
    )
    add_issue(issues, checkpoint["missing_shards"] == [], "one or more safetensors shards are missing")
    add_issue(
        issues,
        checkpoint["payload_bytes_present"] >= args.expected_min_payload_bytes,
        f"checkpoint payload bytes are below {args.expected_min_payload_bytes}",
    )

    config_path = Path(args.config_path or args.checkpoint_dir)
    config_report = {
        "config_path": str(config_path),
        "config_json": str(config_path / "config.json") if config_path.is_dir() else str(config_path),
        "config_json_exists": (config_path / "config.json").exists() if config_path.is_dir() else config_path.exists(),
    }
    add_issue(issues, config_report["config_json_exists"], "config.json is missing or config path is not local")

    runtime = runtime_report(args.reference_device, args.candidate_device, args.npu_smi_cmd, args.require_free_hbm_mb)
    add_issue(issues, runtime["import_errors"] == [], f"runtime import/probe errors: {runtime['import_errors']}")
    add_issue(
        issues,
        version_at_least(runtime.get("transformers_version"), (5, 12, 0)),
        "transformers>=5.12.0 is required",
    )
    validate_device(runtime, args.reference_device, issues)
    validate_device(runtime, args.candidate_device, issues)

    if args.require_free_hbm_mb > 0:
        npu_smi = runtime.get("npu_smi") or {}
        add_issue(issues, npu_smi.get("returncode") == 0, "npu-smi command failed")
        add_issue(issues, bool(npu_smi.get("hbm")), "npu-smi output did not include parseable HBM usage")
        add_issue(
            issues,
            npu_smi.get("devices_with_required_free_hbm", 0) >= 1,
            f"no NPU device has >= {args.require_free_hbm_mb} free HBM MB",
        )

    checkpoint_disk = disk_report(args.checkpoint_dir)
    output_disk = disk_report(args.output_json)
    required_free_disk_bytes = int(args.require_free_disk_gb * (1024**3))
    if required_free_disk_bytes > 0:
        add_issue(
            issues,
            checkpoint_disk["free_bytes"] >= required_free_disk_bytes,
            f"checkpoint filesystem has less than {args.require_free_disk_gb:g} GiB free",
        )
        add_issue(
            issues,
            output_disk["free_bytes"] >= required_free_disk_bytes,
            f"output filesystem has less than {args.require_free_disk_gb:g} GiB free",
        )

    report = {
        "passed": not issues,
        "issues": issues,
        "checkpoint": checkpoint,
        "config": config_report,
        "runtime": runtime,
        "disk": {
            "checkpoint": checkpoint_disk,
            "output": output_disk,
            "required_free_disk_gb": args.require_free_disk_gb,
        },
        "requirements": {
            "expected_shards": args.expected_shards,
            "expected_min_weight_map_keys": args.expected_min_weight_map_keys,
            "expected_min_payload_bytes": args.expected_min_payload_bytes,
            "require_free_hbm_mb": args.require_free_hbm_mb,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
