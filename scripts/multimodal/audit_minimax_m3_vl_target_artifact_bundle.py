#!/usr/bin/env python3
"""Audit a returned MiniMax M3 VL target-machine precision artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_ROOT_NAME = "minimax_m3_vl_target_precision_suite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-root", type=Path, help="Extracted target precision suite directory.")
    source.add_argument("--artifact-tar", type=Path, help="Returned .tgz/.tar.gz artifact bundle.")
    parser.add_argument("--expected-revision", default=os.environ.get("MINIMAX_M3_REFERENCE_REVISION", ""))
    parser.add_argument("--expected-veomni-commit", default="")
    parser.add_argument("--require-target-toy", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def find_suite_root(base: Path) -> Path | None:
    if base.is_dir() and base.name == DEFAULT_ROOT_NAME:
        return base
    if not base.exists():
        return None
    matches = [path for path in base.rglob(DEFAULT_ROOT_NAME) if path.is_dir()]
    return matches[0] if matches else None


def extract_tar(path: Path) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="minimax_m3_vl_artifacts_")
    target_dir = Path(temp_dir.name).resolve()
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            if not str(destination).startswith(str(target_dir) + os.sep):
                raise RuntimeError(f"unsafe tar member path: {member.name}")
        archive.extractall(temp_dir.name)
    return temp_dir


def parse_metadata(path: Path, issues: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        issues.append(f"missing run metadata: {path}")
        return metadata
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(" ")
        metadata[key] = value.strip()
    for key in ("veomni_commit", "minimax_revision", "checkpoint_dir", "generated_at"):
        add_issue(issues, bool(metadata.get(key)), f"run metadata missing {key}")
    return metadata


def resolve_manifest_path(base: Path, suite_root: Path, recorded: str) -> Path | None:
    recorded_path = Path(recorded)
    candidates = []
    if recorded_path.is_absolute():
        candidates.append(recorded_path)
    for parent in (base, Path.cwd(), suite_root, *suite_root.parents):
        candidates.append(parent / recorded_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def audit_manifest(base: Path, suite_root: Path, issues: list[str]) -> dict[str, Any]:
    manifest_path = suite_root / "artifact_manifest.sha256"
    report = {"path": str(manifest_path), "checked": 0, "missing": [], "mismatched": []}
    if not manifest_path.exists():
        issues.append(f"missing artifact manifest: {manifest_path}")
        return report
    for line in manifest_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        digest, _, recorded = line.partition("  ")
        if not recorded:
            digest, _, recorded = line.partition(" ")
        recorded = recorded.strip()
        if Path(recorded).name == "artifact_manifest.sha256":
            issues.append("artifact manifest must not include itself")
            continue
        actual_path = resolve_manifest_path(base, suite_root, recorded)
        if actual_path is None:
            report["missing"].append(recorded)
            continue
        actual = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        report["checked"] += 1
        if actual != digest:
            report["mismatched"].append(recorded)
    add_issue(issues, report["checked"] > 0, "artifact manifest checked no files")
    add_issue(issues, not report["missing"], f"artifact manifest missing files: {report['missing']}")
    add_issue(issues, not report["mismatched"], f"artifact manifest SHA mismatches: {report['mismatched']}")
    return report


def parse_json_from_log(path: Path, issues: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing {label} log: {path}")
        return {}
    text = path.read_text(errors="replace")
    if not text.strip():
        issues.append(f"empty {label} log: {path}")
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        issues.append(f"{label} log has no JSON runtime report: {path}")
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        issues.append(f"{label} log JSON parse failed: {exc}")
        return {}


def audit_final_jsons(suite_root: Path, issues: list[str], expected_revision: str) -> dict[str, Any]:
    paths = {
        "final": suite_root / "final_precision_audit.json",
        "full_preflight": suite_root / "full_checkpoint" / "full_checkpoint_preflight.json",
        "full_forward": suite_root / "full_checkpoint" / "full_checkpoint_forward.json",
        "full_audit": suite_root / "full_checkpoint" / "full_checkpoint_audit.json",
        "multicard": suite_root / "multicard" / "multicard_parity_summary.json",
    }
    data: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.exists():
            issues.append(f"missing required artifact {name}: {path}")
            data[name] = {}
            continue
        data[name] = load_json(path)

    final = data["final"]
    add_issue(issues, final.get("passed") is True, "final audit did not pass")
    add_issue(issues, final.get("full_checkpoint_preflight_passed") is True, "full preflight not proven in final audit")
    add_issue(issues, final.get("full_checkpoint_forward_passed") is True, "full forward not proven in final audit")
    add_issue(issues, final.get("multicard_passed") is True, "multicard not proven in final audit")

    preflight = data["full_preflight"]
    official_reference = preflight.get("official_reference") or {}
    add_issue(issues, preflight.get("passed") is True, "full preflight artifact did not pass")
    add_issue(issues, preflight.get("issues") == [], "full preflight artifact has issues")
    add_issue(
        issues,
        official_reference.get("model_id") == "MiniMaxAI/MiniMax-M3",
        "preflight official reference model_id is not MiniMaxAI/MiniMax-M3",
    )
    if expected_revision:
        add_issue(
            issues,
            official_reference.get("revision") == expected_revision,
            "preflight official reference revision does not match expected revision",
        )
    checkpoint = preflight.get("checkpoint") or {}
    add_issue(issues, checkpoint.get("selected_shard_count", 0) >= 59, "preflight did not see all 59 shards")
    add_issue(issues, checkpoint.get("payload_bytes_present", 0) > 800_000_000_000, "preflight payload bytes too low")

    forward = data["full_forward"]
    payload = forward.get("payload") or {}
    forward_section = forward.get("forward") or {}
    add_issue(issues, forward.get("passed") is True, "full forward artifact did not pass")
    add_issue(issues, forward.get("full_checkpoint_load_executed") is True, "full checkpoint load was not executed")
    add_issue(issues, forward.get("failed") == [], "full forward artifact has failed checks")
    add_issue(issues, payload.get("public_keys_read", 0) >= 20_000, "full forward did not read full public weight map")
    add_issue(issues, payload.get("payload_bytes_read", 0) > 800_000_000_000, "full forward payload bytes too low")
    state_load = forward_section.get("state_dict_load") or {}
    add_issue(issues, state_load.get("strict") is True, "full forward state_dict load was not strict")
    add_issue(issues, forward_section.get("failed") == [], "full forward section has failed checks")

    full_audit = data["full_audit"]
    add_issue(issues, full_audit.get("passed") is True, "full checkpoint audit did not pass")
    add_issue(issues, full_audit.get("full_checkpoint_forward_passed") is True, "full checkpoint audit lacks forward pass")

    multicard = data["multicard"]
    add_issue(issues, multicard.get("passed") is True, "multicard summary did not pass")
    for name in ("preflight", "dummy_forward", "e2e_align"):
        item = multicard.get(name) or {}
        log_value = item.get("log")
        if isinstance(log_value, str):
            log_path = Path(log_value)
            if not log_path.is_absolute():
                log_path = (suite_root / "multicard" / log_path.name).resolve()
            add_issue(issues, log_path.exists(), f"missing multicard {name} log: {log_path}")
            if log_path.exists():
                add_issue(issues, log_path.read_text(errors="replace").strip() != "", f"empty multicard {name} log")

    runtime = parse_json_from_log(suite_root / "multicard" / "preflight.log", issues, "multicard preflight")
    device_count = runtime.get("device_count")
    min_devices = runtime.get("min_devices")
    if isinstance(device_count, int) and isinstance(min_devices, int):
        add_issue(issues, device_count >= min_devices, "multicard preflight device_count below min_devices")
    add_issue(issues, runtime.get("errors") == [], f"multicard preflight errors present: {runtime.get('errors')}")
    if runtime.get("device_type") == "npu":
        add_issue(issues, runtime.get("torch_npu_version") is not None, "multicard NPU missing torch_npu_version")
        ascend_env = runtime.get("ascend_env") or {}
        add_issue(
            issues,
            ascend_env.get("ASCEND_RT_VISIBLE_DEVICES") is not None
            or ascend_env.get("ASCEND_VISIBLE_DEVICES") is not None,
            "multicard NPU missing visible Ascend device env",
        )
    required_free_hbm_mb = runtime.get("required_free_hbm_mb", 0)
    if isinstance(required_free_hbm_mb, int) and required_free_hbm_mb > 0:
        npu_smi = runtime.get("npu_smi") or {}
        add_issue(issues, npu_smi.get("returncode") == 0, "multicard npu-smi preflight failed")
        if isinstance(min_devices, int):
            add_issue(
                issues,
                npu_smi.get("devices_with_required_free_hbm", 0) >= min_devices,
                "not enough NPU devices with required free HBM",
            )
    return {name: str(path) for name, path in paths.items()}


def audit_target_toy(suite_root: Path, issues: list[str], required: bool) -> dict[str, Any]:
    path = suite_root / "toy" / "toy_hf_cpu_veomni_npu_parity_target.json"
    if not path.exists():
        if required:
            issues.append(f"missing target toy parity artifact: {path}")
        return {"path": str(path), "present": False}
    data = load_json(path)
    add_issue(issues, data.get("passed") is True, "target toy parity did not pass")
    add_issue(issues, data.get("failed") == [], "target toy parity has failed checks")
    add_issue(issues, data.get("reference_device") == "cpu", "target toy parity reference_device is not cpu")
    add_issue(issues, str(data.get("candidate_device", "")).startswith("npu"), "target toy parity candidate is not npu")
    return {"path": str(path), "present": True}


def main() -> None:
    args = parse_args()
    issues: list[str] = []
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    base = args.artifact_root
    if args.artifact_tar is not None:
        temp_dir = extract_tar(args.artifact_tar)
        base = Path(temp_dir.name)

    assert base is not None
    suite_root = find_suite_root(base)
    if suite_root is None:
        raise SystemExit(f"could not find {DEFAULT_ROOT_NAME} under {base}")

    metadata = parse_metadata(suite_root / "run_metadata.txt", issues)
    if args.expected_revision:
        add_issue(
            issues,
            metadata.get("minimax_revision") == args.expected_revision,
            "run metadata minimax_revision does not match expected revision",
        )
    if args.expected_veomni_commit:
        add_issue(
            issues,
            metadata.get("veomni_commit") == args.expected_veomni_commit,
            "run metadata veomni_commit does not match expected commit",
        )

    report = {
        "passed": False,
        "suite_root": str(suite_root),
        "metadata": metadata,
        "manifest": audit_manifest(base, suite_root, issues),
        "required_artifacts": audit_final_jsons(suite_root, issues, args.expected_revision),
        "target_toy": audit_target_toy(suite_root, issues, args.require_target_toy),
        "issues": issues,
    }
    report["passed"] = not issues
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if temp_dir is not None:
        temp_dir.cleanup()
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
