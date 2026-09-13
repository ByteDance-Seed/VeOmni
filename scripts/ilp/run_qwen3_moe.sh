#!/usr/bin/env bash
# Run a portable Qwen3-MoE FSDP2 baseline or ILP experiment on Ascend NPU.
#
# Required:
#   MODEL_PATH=/path/to/Qwen3-30B-A3B
#   TRAIN_PATH=/path/to/train.parquet  # or another VeOmni dataset source
#
# Examples:
#   MODE=baseline NPROC_PER_NODE=16 EP_SIZE=16 bash scripts/ilp/run_qwen3_moe.sh
#   MODE=ilp WINDOW_SIZE=20 NPROC_PER_NODE=16 EP_SIZE=16 bash scripts/ilp/run_qwen3_moe.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-MoE checkpoint directory.}"
: "${TRAIN_PATH:?Set TRAIN_PATH to the training dataset.}"

MODE="${MODE:-ilp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
EP_SIZE="${EP_SIZE:-${NPROC_PER_NODE}}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${NPROC_PER_NODE}}"
MAX_STEPS="${MAX_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
WINDOW_SIZE="${WINDOW_SIZE:-0}"
PROFILE_ENABLE="${PROFILE_ENABLE:-false}"
PROFILE_START_STEP="${PROFILE_START_STEP:-2}"
PROFILE_END_STEP="${PROFILE_END_STEP:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen3_moe_${MODE}}"
TRACE_DIR="${TRACE_DIR:-${REPO_ROOT}/trace/qwen3_moe_${MODE}}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/log_qwen3_moe_${MODE}.txt}"

case "${MODE}" in
  baseline)
    ILP_ENABLE=false
    ;;
  ilp)
    ILP_ENABLE=true
    ;;
  *)
    echo "MODE must be 'baseline' or 'ilp', got: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${EP_SIZE}" != "${NPROC_PER_NODE}" && "${ILP_ENABLE}" == "true" ]]; then
  echo "ILP currently requires EP_SIZE == NPROC_PER_NODE." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${TRACE_DIR}" "$(dirname -- "${LOG_FILE}")"
cd "${REPO_ROOT}"

torchrun \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  tasks/train_text.py \
  configs/text/qwen3-moe.yaml \
  --model.model_path "${MODEL_PATH}" \
  --model.ops_implementation.moe_implementation fused_npu \
  --data.train_path "${TRAIN_PATH}" \
  --data.max_seq_len "${MAX_SEQ_LEN}" \
  --train.accelerator.ep_size "${EP_SIZE}" \
  --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
  --train.init_device meta \
  --train.gradient_checkpointing.enable true \
  --train.gradient_checkpointing.enable_reentrant false \
  --train.inter_layer_replay.enable "${ILP_ENABLE}" \
  --train.inter_layer_replay.current_layer -1 \
  --train.inter_layer_replay.window_size "${WINDOW_SIZE}" \
  --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --train.max_steps "${MAX_STEPS}" \
  --train.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --train.checkpoint.output_dir "${OUTPUT_DIR}" \
  --train.checkpoint.save_steps 0 \
  --train.checkpoint.save_epochs 0 \
  --train.checkpoint.save_hf_weights false \
  --train.profile.enable "${PROFILE_ENABLE}" \
  --train.profile.start_step "${PROFILE_START_STEP}" \
  --train.profile.end_step "${PROFILE_END_STEP}" \
  --train.profile.trace_dir "${TRACE_DIR}" \
  --train.profile.profile_memory false \
  --train.profile.with_stack false \
  --train.profile.rank0_only true \
  "$@" 2>&1 | tee "${LOG_FILE}"
