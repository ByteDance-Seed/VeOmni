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
import sys

min_devices = int(sys.argv[1])

import torch
import transformers

version_parts = tuple(int(part) for part in re.findall(r"\d+", transformers.__version__)[:3])
if version_parts < (5, 12, 0):
    raise SystemExit(f"transformers>=5.12.0 required, got {transformers.__version__}")

torch_npu_version = None
if hasattr(torch, "npu"):
    try:
        import torch_npu  # noqa: F401

        torch_npu_version = getattr(torch_npu, "__version__", None)
    except Exception:
        torch_npu_version = None

if hasattr(torch, "cuda") and torch.cuda.is_available():
    device_type = "cuda"
    device_count = torch.cuda.device_count()
elif hasattr(torch, "npu") and torch.npu.is_available():
    device_type = "npu"
    device_count = torch.npu.device_count()
else:
    device_type = None
    device_count = 0

report = {
    "device_type": device_type,
    "device_count": device_count,
    "min_devices": min_devices,
    "torch_version": torch.__version__,
    "torch_npu_version": torch_npu_version,
    "transformers_version": transformers.__version__,
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
}
print(json.dumps(report, indent=2, sort_keys=True))
if device_count < min_devices:
    raise SystemExit(f"requires at least {min_devices} accelerator devices, got {device_count}")
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
