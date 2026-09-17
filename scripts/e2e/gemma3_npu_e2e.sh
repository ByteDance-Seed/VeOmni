#!/bin/bash
# ==============================================================================
# Gemma 3 NPU End-to-End Training Script
#
# Runs a short Gemma 3 training on Ascend NPU to verify the NPU migration is
# functional: model builds, forward/backward works, loss decreases, and no
# device-incompatible kernels fire.
#
# Usage:
#   bash scripts/e2e/gemma3_npu_e2e.sh [CONFIG] [EXTRA_ARGS...]
#
# Prerequisites:
#   - torch_npu installed and ASCEND environment configured
#   - tulu-3-sft-mixture dataset downloaded (see configs/text/gemma3_npu.yaml)
#   - google/gemma-3-270m model weights cached locally
#
# Environment overrides:
#   NNODES (default 1), NPROC_PER_NODE (auto-detected from NPU count)
#   MAX_STEPS (default 20) — short smoke run; set higher for full benchmark
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/text/gemma3_npu.yaml}"
MAX_STEPS="${MAX_STEPS:-20}"

# --- NPU device count ---------------------------------------------------------
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-$(ls -1 /dev/davinci* 2>/dev/null | grep -vc 'davinci_manager')}"
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"

if [[ "${NNODES}" == "1" ]]; then
  RDZV_ARGS="--standalone"
else
  RDZV_ARGS="--rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

echo "================================================================"
echo " Gemma 3 NPU E2E Training"
echo "   Config:        ${CONFIG}"
echo "   Max steps:     ${MAX_STEPS}"
echo "   NPU devices:   ${NPROC_PER_NODE}"
echo "   Nodes:         ${NNODES}"
echo "================================================================"

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  ${RDZV_ARGS} \
  tasks/train_text.py \
  --config "${CONFIG}" \
  --train.max_steps "${MAX_STEPS}" \
  "${@:2}" 2>&1 | tee gemma3_npu_e2e.log
