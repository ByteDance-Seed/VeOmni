#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the target-machine MiniMax M3 VL precision evidence suite.

This suite orchestrates:
  1. Full public-checkpoint preflight and forward parity.
  2. Multi-card SP/EP/FSDP2 parity.
  3. Final strict artifact audit requiring all target-machine artifacts.

Required:
  --checkpoint-dir PATH       Local MiniMaxAI/MiniMax-M3 snapshot with all 59 safetensors shards.

Common options:
  --output-root PATH          Root directory for suite artifacts.
                              Default: docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite
  --config-path PATH          Defaults to --checkpoint-dir.
  --reference-device DEVICE   cpu, cuda, or npu. Default: cpu.
  --candidate-device DEVICE   cpu, cuda, or npu. Default: npu.
  --torch-dtype DTYPE         float32, float16, or bfloat16. Default: bfloat16.
  --prompt-kind KIND          text or multimodal. Default: multimodal.
  --prompt-ids IDS            Comma-separated ids for text prompts. Default: 1,1209,318,257,1332.
  --seq-len N                 Multimodal synthetic prompt length. Default: 10.
  --top-k N                   Last-token top-k ids to compare. Default: 8.
  --max-new-tokens N          Greedy decode tokens to compare. Default: 8.
  --atol VALUE                Forward absolute tolerance. Default: delegated to full-checkpoint runner.
  --rtol VALUE                Forward relative tolerance. Default: delegated to full-checkpoint runner.
  --require-free-disk-gb N    Require N GiB free before full checkpoint work. Default: 0.
  --require-free-hbm-mb N     Require free HBM MB for NPU preflight gates. Default: 0.
  --npu-smi-cmd CMD           Command that prints npu-smi info for preflight evidence.
  --min-devices N             Minimum devices for multi-card parity. Default: 8.
  --python-cmd CMD            Python executable. Default: python3.
  --dry-run                   Print commands without executing them.
  -h, --help                  Show this help.

Example:
  scripts/multimodal/run_minimax_m3_vl_precision_suite.sh \
    --checkpoint-dir /data/checkpoints/MiniMax-M3 \
    --reference-device cpu \
    --candidate-device npu \
    --require-free-disk-gb 50 \
    --require-free-hbm-mb 4096 \
    --npu-smi-cmd 'sudo -n /usr/local/sbin/npu-smi info'
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_DIR=""
CONFIG_PATH=""
OUTPUT_ROOT="docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite"
REFERENCE_DEVICE="cpu"
CANDIDATE_DEVICE="npu"
TORCH_DTYPE="bfloat16"
PROMPT_KIND="multimodal"
PROMPT_IDS="1,1209,318,257,1332"
SEQ_LEN="10"
TOP_K="8"
MAX_NEW_TOKENS="8"
ATOL=""
RTOL=""
REQUIRE_FREE_DISK_GB="0"
REQUIRE_FREE_HBM_MB="0"
NPU_SMI_CMD="${MINIMAX_NPU_SMI_CMD:-}"
MIN_DEVICES="8"
PYTHON_CMD="python3"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --reference-device)
      REFERENCE_DEVICE="$2"
      shift 2
      ;;
    --candidate-device)
      CANDIDATE_DEVICE="$2"
      shift 2
      ;;
    --torch-dtype)
      TORCH_DTYPE="$2"
      shift 2
      ;;
    --prompt-kind)
      PROMPT_KIND="$2"
      shift 2
      ;;
    --prompt-ids)
      PROMPT_IDS="$2"
      shift 2
      ;;
    --seq-len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --atol)
      ATOL="$2"
      shift 2
      ;;
    --rtol)
      RTOL="$2"
      shift 2
      ;;
    --require-free-disk-gb)
      REQUIRE_FREE_DISK_GB="$2"
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
    --min-devices)
      MIN_DEVICES="$2"
      shift 2
      ;;
    --python-cmd)
      PYTHON_CMD="$2"
      shift 2
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

if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "--checkpoint-dir is required" >&2
  usage >&2
  exit 2
fi

CONFIG_PATH="${CONFIG_PATH:-$CHECKPOINT_DIR}"
FULL_DIR="$OUTPUT_ROOT/full_checkpoint"
MULTICARD_DIR="$OUTPUT_ROOT/multicard"
FINAL_AUDIT_JSON="$OUTPUT_ROOT/final_precision_audit.json"
FULL_PREFLIGHT_JSON="$FULL_DIR/full_checkpoint_preflight.json"
FULL_FORWARD_JSON="$FULL_DIR/full_checkpoint_forward.json"
FULL_AUDIT_JSON="$FULL_DIR/full_checkpoint_audit.json"
MULTICARD_JSON="$MULTICARD_DIR/multicard_parity_summary.json"

mkdir -p "$FULL_DIR" "$MULTICARD_DIR"

full_cmd=(
  scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh
  --checkpoint-dir "$CHECKPOINT_DIR"
  --config-path "$CONFIG_PATH"
  --reference-device "$REFERENCE_DEVICE"
  --candidate-device "$CANDIDATE_DEVICE"
  --torch-dtype "$TORCH_DTYPE"
  --prompt-kind "$PROMPT_KIND"
  --prompt-ids "$PROMPT_IDS"
  --seq-len "$SEQ_LEN"
  --top-k "$TOP_K"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --require-free-disk-gb "$REQUIRE_FREE_DISK_GB"
  --require-free-hbm-mb "$REQUIRE_FREE_HBM_MB"
  --preflight-json "$FULL_PREFLIGHT_JSON"
  --output-json "$FULL_FORWARD_JSON"
  --audit-json "$FULL_AUDIT_JSON"
  --python-cmd "$PYTHON_CMD"
)
multicard_cmd=(
  scripts/multimodal/run_minimax_m3_vl_multicard_parity.sh
  --output-dir "$MULTICARD_DIR"
  --min-devices "$MIN_DEVICES"
  --require-free-hbm-mb "$REQUIRE_FREE_HBM_MB"
  --python-cmd "$PYTHON_CMD"
)
final_audit_cmd=(
  "$PYTHON_CMD" scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py
  --require-full-checkpoint-preflight
  --full-preflight-json "$FULL_PREFLIGHT_JSON"
  --require-full-checkpoint-forward
  --full-forward-json "$FULL_FORWARD_JSON"
  --require-multicard
  --multicard-json "$MULTICARD_JSON"
  --output-json "$FINAL_AUDIT_JSON"
)
if [[ -n "$NPU_SMI_CMD" ]]; then
  full_cmd+=(--npu-smi-cmd "$NPU_SMI_CMD")
  multicard_cmd+=(--npu-smi-cmd "$NPU_SMI_CMD")
fi
if [[ -n "$ATOL" ]]; then
  full_cmd+=(--atol "$ATOL")
fi
if [[ -n "$RTOL" ]]; then
  full_cmd+=(--rtol "$RTOL")
fi

print_command() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

print_command "Full checkpoint command" "${full_cmd[@]}"
print_command "Multi-card command" "${multicard_cmd[@]}"
print_command "Final strict audit command" "${final_audit_cmd[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${full_cmd[@]}"
"${multicard_cmd[@]}"
"${final_audit_cmd[@]}"
