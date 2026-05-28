#!/bin/bash
# Qwen3-VL-MoE 30B-A3B WS-PUSH fused QKV + Ulysses a2a, dummy data.
# sp_size is capped at text_config.num_key_value_heads=4 (see yaml).
#
# Usage:
#   bash scripts/multimodal/train_qwen3_vl_moe_ws_push.sh
#   NPROC_PER_NODE=8 bash scripts/multimodal/train_qwen3_vl_moe_ws_push.sh --train.global_batch_size 2
set -x
set -o pipefail

export NPROC_PER_NODE=${NPROC_PER_NODE:=4}
export NNODES=${NNODES:=1}

export NCCL_DEBUG=${NCCL_DEBUG:=WARN}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:=1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:=false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bash train.sh \
  tasks/train_qwen3_vl_moe_dummy.py \
  configs/multimodal/qwen3_vl/qwen3_vl_moe_ws_push_dummy.yaml \
  "$@"
