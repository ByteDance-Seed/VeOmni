#!/usr/bin/env bash
# Example inference invocation for Qwen-Image (T2I) using scripts/inference/infer_omni.py.
#
# Usage:
#   bash scripts/inference/infer_qwen_image.sh
#
# Override anything via environment variables:
#   MODEL_PATH=/path/to/Qwen-Image OUTPUT_DIR=./out bash scripts/inference/infer_qwen_image.sh
#
# To swap in a VeOmni-fine-tuned transformer, set TRANSFORMER_PATH to the
# checkpoint directory (containing config.json + model.safetensors[.index.json]):
#   TRANSFORMER_PATH=./qwen-image-sft/checkpoints/global_step_500/hf_ckpt \
#       bash scripts/inference/infer_qwen_image.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image}"
OUTPUT_DIR="${OUTPUT_DIR:-./inference_outputs/qwen_image}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-}"

PROMPTS=(
    "A cinematic close-up portrait of a corgi wearing a tiny astronaut helmet, studio lighting, shallow depth of field."
    "Anime-style illustration of a cyberpunk skyline at dawn, neon reflections on wet streets, ultra-detailed."
)

EXTRA_ARGS=()
if [[ -n "${TRANSFORMER_PATH}" ]]; then
    EXTRA_ARGS+=(--transformer_path "${TRANSFORMER_PATH}")
fi

python "${REPO_ROOT}/scripts/inference/infer_omni.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompts "${PROMPTS[@]}" \
    --num_inference_steps 50 \
    --num_images_per_prompt 1 \
    --height 1024 \
    --width 1024 \
    --true_cfg_scale 4.0 \
    --seed 42 \
    --dtype bfloat16 \
    --enable_cpu_offload \
    "${EXTRA_ARGS[@]}"
