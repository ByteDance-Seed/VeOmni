#!/bin/bash
# Simple FSDP2 XPU test - standalone, no trainer dependencies

set -e

echo "=========================================="
echo "VeOmni FSDP2 Standalone Test on Intel XPU"
echo "=========================================="
echo "GPUs: 2 (XPU devices 0,1)"
echo "Model: Qwen2.5-0.5B-Instruct (sdpa attention)"
echo ""

# XPU environment variables
export ZE_AFFINITY_MASK="0,1"
export CCL_ATL_SHM=1
export CCL_BUFFER_CACHE=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0
export RAY_NUM_PRESTART_PYTHON_WORKERS=0

# Run with torchrun (2 XPU GPUs)
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master-port=4321 \
    tests/special_xpu/test_fsdp2_simple_xpu.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 2 \
    --seq-len 128 \
    --steps 5

echo ""
echo "=========================================="
echo "VeOmni FSDP2 XPU test completed!"
echo "=========================================="
