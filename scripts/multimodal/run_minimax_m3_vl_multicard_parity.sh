#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run MiniMax M3 VL multi-card SP/EP/FSDP2 parity gates on target hardware.

This launcher is intended for GPU/NPU target machines. It fails if the local
runtime cannot expose enough devices, then runs:
  1. FSDP2 asymmetric multimodal dummy-forward hang/gradient gate.
  2. MiniMax SP/EP/FSDP2 e2e alignment with pytest --runxfail.

Options:
  --output-dir PATH       Directory for logs and multicard_parity_summary.json.
                          Default: docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity
  --min-devices N         Minimum accelerator device count. Default: 8.
  --require-free-hbm-mb N Require at least N free HBM MB on each counted NPU.
                          Default: 0, disabled.
  --npu-smi-cmd CMD       Command that prints npu-smi info for preflight evidence.
                          Default: auto-detect npu-smi info.
  --python-cmd CMD        Python executable. Default: python3.
  --skip-dummy-forward    Skip asymmetric dummy-forward gate.
  --skip-e2e-align        Skip SP/EP/FSDP2 e2e alignment gate.
  --dry-run               Print commands without executing them.
  -h, --help              Show this help.

Example:
  scripts/multimodal/run_minimax_m3_vl_multicard_parity.sh \
    --min-devices 8 \
    --output-dir docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity"
MIN_DEVICES="8"
REQUIRE_FREE_HBM_MB="0"
NPU_SMI_CMD="${MINIMAX_NPU_SMI_CMD:-}"
PYTHON_CMD="python3"
RUN_DUMMY=1
RUN_E2E=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --min-devices)
      MIN_DEVICES="$2"
      shift 2
      ;;
    --require-free-hbm-mb)
      REQUIRE_FREE_HBM_MB="$2"
      shift 2
      ;;
    --npu-smi-cmd)
      NPU_SMI_CMD="$2"
      shift 2
      ;;
    --python-cmd)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --skip-dummy-forward)
      RUN_DUMMY=0
      shift
      ;;
    --skip-e2e-align)
      RUN_E2E=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
export MODELING_BACKEND="${MODELING_BACKEND:-veomni}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

preflight_cmd=(
  "$PYTHON_CMD" -
  "$MIN_DEVICES"
  "$REQUIRE_FREE_HBM_MB"
  "$NPU_SMI_CMD"
)
dummy_cmd=(
  "$PYTHON_CMD" -m pytest
  'tests/distributed/test_dummy_forward.py::test_asymmetric_forward_vlm[minimax_m3_vl]'
  -q -s
)
e2e_cmd=(
  "$PYTHON_CMD" -m pytest
  tests/e2e/test_e2e_parallel.py::test_minimax_m3_vl_parallel_align
  --runxfail
  -q -s
)

print_command() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

print_command "Preflight command" "${preflight_cmd[@]}"
if [[ "$RUN_DUMMY" -eq 1 ]]; then
  print_command "Dummy-forward command" "${dummy_cmd[@]}"
fi
if [[ "$RUN_E2E" -eq 1 ]]; then
  print_command "E2E align command" "${e2e_cmd[@]}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

run_status=0
preflight_log="$OUTPUT_DIR/preflight.log"
dummy_log="$OUTPUT_DIR/dummy_forward.log"
e2e_log="$OUTPUT_DIR/e2e_align.log"
summary_json="$OUTPUT_DIR/multicard_parity_summary.json"

set +e
"${preflight_cmd[@]}" <<'PY' 2>&1 | tee "$preflight_log"
import json
import os
import re
import shutil
import subprocess
import sys

min_devices = int(sys.argv[1])
required_free_hbm_mb = int(sys.argv[2])
npu_smi_cmd = sys.argv[3] or ""

def run_command(command):
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

def default_npu_smi_command():
    for candidate in ("npu-smi", "/usr/local/sbin/npu-smi", "/usr/local/bin/npu-smi"):
        if shutil.which(candidate) or os.path.exists(candidate):
            return f"{candidate} info"
    return ""

def parse_npu_hbm(output):
    devices = []
    current_device = None
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

npu_smi_report = run_command(npu_smi_cmd or default_npu_smi_command())
if npu_smi_report is not None:
    npu_smi_report["hbm"] = parse_npu_hbm(npu_smi_report.get("output_excerpt", ""))
    npu_smi_report["required_free_hbm_mb"] = required_free_hbm_mb
    npu_smi_report["devices_with_required_free_hbm"] = sum(
        1 for device in npu_smi_report["hbm"] if device["free_mb"] >= required_free_hbm_mb
    )

errors = []

try:
    import torch
except Exception as exc:
    torch = None
    torch_version = None
    errors.append(f"torch import failed: {exc!r}")
else:
    torch_version = torch.__version__

try:
    import transformers
except Exception as exc:
    transformers = None
    transformers_version = None
    errors.append(f"transformers import failed: {exc!r}")
else:
    transformers_version = transformers.__version__
    version_parts = tuple(int(part) for part in re.findall(r"\d+", transformers_version)[:3])
    if version_parts < (5, 12, 0):
        errors.append(f"transformers>=5.12.0 required, got {transformers_version}")

torch_npu_version = None
if torch is not None and hasattr(torch, "npu"):
    try:
        import torch_npu  # noqa: F401

        torch_npu_version = getattr(torch_npu, "__version__", None)
    except Exception as exc:
        torch_npu_version = None
        errors.append(f"torch_npu import failed: {exc!r}")

if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
    device_type = "cuda"
    device_count = torch.cuda.device_count()
elif torch is not None and hasattr(torch, "npu") and torch.npu.is_available():
    device_type = "npu"
    device_count = torch.npu.device_count()
else:
    device_type = None
    device_count = 0

if device_count < min_devices:
    errors.append(f"requires at least {min_devices} accelerator devices, got {device_count}")
if required_free_hbm_mb > 0:
    if npu_smi_report is None:
        errors.append("npu-smi command is required when --require-free-hbm-mb is set")
    elif npu_smi_report.get("returncode") != 0:
        errors.append(f"npu-smi command failed with return code {npu_smi_report.get('returncode')}")
    elif not npu_smi_report.get("hbm"):
        errors.append("npu-smi output did not include parseable HBM usage")
    elif npu_smi_report.get("devices_with_required_free_hbm", 0) < min_devices:
        errors.append(
            f"requires at least {min_devices} NPU devices with >= {required_free_hbm_mb} free HBM MB, "
            f"got {npu_smi_report.get('devices_with_required_free_hbm', 0)}"
        )

report = {
    "device_type": device_type,
    "device_count": device_count,
    "min_devices": min_devices,
    "required_free_hbm_mb": required_free_hbm_mb,
    "torch_version": torch_version,
    "torch_npu_version": torch_npu_version,
    "transformers_version": transformers_version,
    "npu_smi": npu_smi_report,
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
    "errors": errors,
}
print(json.dumps(report, indent=2, sort_keys=True))
if errors:
    raise SystemExit("; ".join(errors))
PY
preflight_status=${PIPESTATUS[0]}
set -e
if [[ "$preflight_status" -ne 0 ]]; then
  run_status="$preflight_status"
fi

dummy_status=-1
if [[ "$RUN_DUMMY" -eq 1 && "$run_status" -eq 0 ]]; then
  set +e
  "${dummy_cmd[@]}" 2>&1 | tee "$dummy_log"
  dummy_status=${PIPESTATUS[0]}
  set -e
  if [[ "$dummy_status" -ne 0 ]]; then
    run_status="$dummy_status"
  fi
else
  : >"$dummy_log"
fi

e2e_status=-1
if [[ "$RUN_E2E" -eq 1 && "$run_status" -eq 0 ]]; then
  set +e
  "${e2e_cmd[@]}" 2>&1 | tee "$e2e_log"
  e2e_status=${PIPESTATUS[0]}
  set -e
  if [[ "$e2e_status" -ne 0 ]]; then
    run_status="$e2e_status"
  fi
else
  : >"$e2e_log"
fi

"$PYTHON_CMD" - "$summary_json" "$preflight_status" "$dummy_status" "$e2e_status" "$RUN_DUMMY" "$RUN_E2E" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
preflight_status = int(sys.argv[2])
dummy_status = int(sys.argv[3])
e2e_status = int(sys.argv[4])
run_dummy = bool(int(sys.argv[5]))
run_e2e = bool(int(sys.argv[6]))

def returncode(enabled, value):
    if not enabled or value < 0:
        return None
    return value

payload = {
    "passed": preflight_status == 0 and (not run_dummy or dummy_status == 0) and (not run_e2e or e2e_status == 0),
    "preflight": {"returncode": preflight_status, "log": str(summary_path.with_name("preflight.log"))},
    "dummy_forward": {
        "enabled": run_dummy,
        "returncode": returncode(run_dummy, dummy_status),
        "log": str(summary_path.with_name("dummy_forward.log")),
    },
    "e2e_align": {
        "enabled": run_e2e,
        "returncode": returncode(run_e2e, e2e_status),
        "log": str(summary_path.with_name("e2e_align.log")),
        "runxfail_required": True,
    },
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

exit "$run_status"
