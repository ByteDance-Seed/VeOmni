#!/usr/bin/env bash
# Run baseline and ILP sequentially with identical Qwen3-MoE inputs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RUNNER="${SCRIPT_DIR}/run_qwen3_moe.sh"
WINDOW_SIZE="${WINDOW_SIZE:-0}"
PROFILE_ENABLE="${PROFILE_ENABLE:-true}"

MODE=baseline \
OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen3_moe_baseline}" \
TRACE_DIR="${BASELINE_TRACE_DIR:-${REPO_ROOT}/trace/qwen3_moe_baseline}" \
LOG_FILE="${BASELINE_LOG_FILE:-${REPO_ROOT}/log_qwen3_moe_baseline.txt}" \
PROFILE_ENABLE="${PROFILE_ENABLE}" \
bash "${RUNNER}" "$@"

MODE=ilp \
WINDOW_SIZE="${WINDOW_SIZE}" \
OUTPUT_DIR="${ILP_OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen3_moe_ilp_w${WINDOW_SIZE}}" \
TRACE_DIR="${ILP_TRACE_DIR:-${REPO_ROOT}/trace/qwen3_moe_ilp_w${WINDOW_SIZE}}" \
LOG_FILE="${ILP_LOG_FILE:-${REPO_ROOT}/log_qwen3_moe_ilp_w${WINDOW_SIZE}.txt}" \
PROFILE_ENABLE="${PROFILE_ENABLE}" \
bash "${RUNNER}" "$@"
