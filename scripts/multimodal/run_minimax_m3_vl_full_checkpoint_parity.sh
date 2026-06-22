#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run MiniMax M3 VL full-checkpoint forward parity and strict artifact audit.

Required:
  --checkpoint-dir PATH       Local MiniMaxAI/MiniMax-M3 snapshot with all 59 safetensors shards.

Common options:
  --config-path PATH          Defaults to --checkpoint-dir.
  --reference-device DEVICE   cpu, cuda, or npu. Default: cpu.
  --candidate-device DEVICE   cpu, cuda, or npu. Default: npu.
  --torch-dtype DTYPE         float32, float16, or bfloat16. Default: bfloat16.
  --prompt-kind KIND          text or multimodal. Default: multimodal.
  --prompt-ids IDS            Comma-separated ids for text prompts. Default: 1,1209,318,257,1332.
  --seq-len N                 Multimodal synthetic prompt length. Default: 10.
  --top-k N                   Last-token top-k ids to compare. Default: 8.
  --max-new-tokens N          Greedy decode tokens to compare. Default: 8.
  --atol VALUE                Forward absolute tolerance. Default: 5e-4 for NPU candidate, else 2e-4.
  --rtol VALUE                Forward relative tolerance. Default: 5e-4 for NPU candidate, else 2e-4.
  --output-json PATH          Forward artifact path.
  --audit-json PATH           Strict audit artifact path.
  --dry-run                   Print commands without executing them.

Example:
  scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh \
    --checkpoint-dir /data/checkpoints/MiniMax-M3 \
    --reference-device cpu \
    --candidate-device npu
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_DIR=""
CONFIG_PATH=""
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
OUTPUT_JSON=""
AUDIT_JSON=""
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
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --audit-json)
      AUDIT_JSON="$2"
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

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint dir does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi

CONFIG_PATH="${CONFIG_PATH:-$CHECKPOINT_DIR}"
if [[ "$CANDIDATE_DEVICE" == "npu" ]]; then
  ATOL="${ATOL:-5e-4}"
  RTOL="${RTOL:-5e-4}"
else
  ATOL="${ATOL:-2e-4}"
  RTOL="${RTOL:-2e-4}"
fi

sanitize_device() {
  printf '%s' "$1" | tr ':/' '__'
}

if [[ -z "$OUTPUT_JSON" ]]; then
  REF_TAG="$(sanitize_device "$REFERENCE_DEVICE")"
  CAND_TAG="$(sanitize_device "$CANDIDATE_DEVICE")"
  OUTPUT_JSON="docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_${REF_TAG}_${CAND_TAG}.json"
fi
AUDIT_JSON="${AUDIT_JSON:-docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit_full.json}"

if [[ "$REFERENCE_DEVICE" == "npu" || "$CANDIDATE_DEVICE" == "npu" ]]; then
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  fi
fi

export MODELING_BACKEND="${MODELING_BACKEND:-veomni}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

forward_cmd=(
  python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py
  --checkpoint-dir "$CHECKPOINT_DIR"
  --config-path "$CONFIG_PATH"
  --mode forward
  --reference-device "$REFERENCE_DEVICE"
  --candidate-device "$CANDIDATE_DEVICE"
  --torch-dtype "$TORCH_DTYPE"
  --prompt-kind "$PROMPT_KIND"
  --prompt-ids "$PROMPT_IDS"
  --seq-len "$SEQ_LEN"
  --top-k "$TOP_K"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --atol "$ATOL"
  --rtol "$RTOL"
  --confirm-full-load
  --output-json "$OUTPUT_JSON"
)

audit_cmd=(
  python scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py
  --require-full-checkpoint-forward
  --full-forward-json "$OUTPUT_JSON"
  --output-json "$AUDIT_JSON"
)

print_command() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

print_command "Forward parity command" "${forward_cmd[@]}"
print_command "Audit command" "${audit_cmd[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${forward_cmd[@]}"
"${audit_cmd[@]}"
