#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG_PATH="${CONFIG_PATH:-configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml}"

uv run --no-default-groups \
  --with transformers==5.12.0 \
  --with torch==2.7.1 \
  torchrun --nproc_per_node="${NUM_GPUS}" examples/train_vlm.py \
    --config "${CONFIG_PATH}"
