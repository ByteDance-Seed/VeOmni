#!/bin/bash
# VeOmni trainer e2e SFT smoke test on Intel XPU (2 GPUs)

set -e

# Get the VeOmni root directory
VEOMNI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$VEOMNI_ROOT"

# XPU environment variables
export ZE_AFFINITY_MASK="0,1"
export CCL_ATL_SHM=1
export CCL_BUFFER_CACHE=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0
export RAY_NUM_PRESTART_PYTHON_WORKERS=0

# Model and data setup
MODEL_PATH="${CI_HF_MODELS_DIR:-.}/Qwen/Qwen2.5-0.5B-Instruct"
DATASET_DIR="${CI_DATASET_DIR:-.}"

echo "=========================================="
echo "VeOmni e2e SFT Test on Intel XPU"
echo "=========================================="
echo "Model: Qwen2.5-0.5B-Instruct"
echo "GPUs: 2 (XPU devices 0,1 via ZE_AFFINITY_MASK)"
echo "Config: qwen2_5_xpu.yaml (sdpa attention, fsdp1 trainer smoke)"
echo ""

# Run torchrun with 2 XPU GPUs
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master-port=4321 \
    tasks/train_text.py \
    configs/text/qwen2_5_xpu.yaml \
    --model.model_path "$MODEL_PATH" \
    --data.train_path "$DATASET_DIR/fineweb" \
    --train.checkpoint.output_dir "Qwen2.5-0.5B-Instruct-sft-xpu" \
    --train.accelerator.fsdp_config.fsdp_mode fsdp1 \
    --train.num_train_epochs 1 \
    --train.max_steps 5 \
    --train.wandb.enable false

echo ""
echo "=========================================="
echo "VeOmni XPU e2e test PASSED!"
echo "=========================================="
