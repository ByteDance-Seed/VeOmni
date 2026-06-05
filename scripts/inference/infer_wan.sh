#!/usr/bin/env bash
# Example inference invocation for Wan2.1-T2V (text-to-video) using
# scripts/inference/infer_omni.py.
#
# Wan inference currently requires a newer transformers than the project
# pin (``transformers==5.2.0`` from the ``transformers-stable`` group in
# pyproject.toml). We therefore launch via ``uv run --with`` which installs
# the override into a per-invocation cache under ~/.cache/uv/ and prepends
# that cache to PYTHONPATH — the project's ``.venv`` is never modified
# (``--no-sync`` skips the auto-sync that would otherwise touch it).
# Setting TRANSFORMERS_VERSION lets you swap the override without editing
# the script.
#
# Usage:
#   bash scripts/inference/infer_wan.sh
#
# Override anything via environment variables:
#   MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers OUTPUT_DIR=./out \
#       bash scripts/inference/infer_wan.sh
#
# To attach a VeOmni-trained LoRA adapter, set LORA_PATH to the adapter
# directory (containing adapter_config.json + adapter_model.safetensors):
#   LORA_PATH=./exp/Wan2.1-T2V-1.3B-Diffusers_lora/checkpoints/global_step_200 \
#       bash scripts/inference/infer_wan.sh
#
# For Wan2.1 I2V, see the commented block at the bottom of this file: switch
# the model path and add --input_image; the same infer_omni.py CLI handles it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-./inference_outputs/wan_t2v}"
LORA_PATH="${LORA_PATH:-}"
LORA_WEIGHT="${LORA_WEIGHT:-1.0}"
# Pin a newer transformers just for this run; leave the project .venv at the
# transformers-stable group's pin (5.2.0). Pass an explicit empty string
# (``TRANSFORMERS_VERSION='' bash scripts/inference/infer_wan.sh``) to skip
# the overlay and use whatever is already in .venv. Bare ``-`` (not ``:-``)
# distinguishes "unset" from "set to empty".
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION-5.9.0}"

PROMPTS=(
    "Tom, the mischievous gray cat, is sprawled out on a vibrant red pillow, his body relaxed and his eyes half-closed."
)

# Standard Wan video-quality negative prompt from docs/examples/wan2.1_I2V_1.3B.md.
NEGATIVE_PROMPT="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

EXTRA_ARGS=()
if [[ -n "${LORA_PATH}" ]]; then
    EXTRA_ARGS+=(--lora_path "${LORA_PATH}" --lora_weight "${LORA_WEIGHT}")
fi

# Build the launcher prefix. ``uv run --with PKG --no-sync -- python ...``
# layers PKG on top of the project's .venv without writing into it; falls
# back to plain ``python`` when TRANSFORMERS_VERSION is empty.
if [[ -n "${TRANSFORMERS_VERSION}" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[error] TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION} requires \`uv\` on PATH." >&2
        echo "        Install uv (https://github.com/astral-sh/uv) or set TRANSFORMERS_VERSION=." >&2
        exit 1
    fi
    LAUNCHER=(uv run --project "${REPO_ROOT}" --with "transformers==${TRANSFORMERS_VERSION}" --no-sync -- python)
else
    LAUNCHER=(python)
fi

"${LAUNCHER[@]}" "${REPO_ROOT}/scripts/inference/infer_omni.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompts "${PROMPTS[@]}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --num_inference_steps 50 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --fps 15 \
    --guidance_scale 5.0 \
    --seed 42 \
    --dtype bfloat16 \
    --enable_cpu_offload \
    "${EXTRA_ARGS[@]}"

# ──────────────────────────────────────────────────────────────────────────────
# I2V (image-to-video) variant — uncomment to use, requires --input_image:
#
# "${LAUNCHER[@]}" "${REPO_ROOT}/scripts/inference/infer_omni.py" \
#     --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
#     --output_dir "${OUTPUT_DIR%/}/_i2v" \
#     --prompts "the cat slowly stretches and yawns" \
#     --negative_prompt "${NEGATIVE_PROMPT}" \
#     --input_image ./first_frame.png \
#     --num_inference_steps 50 \
#     --height 480 --width 832 --num_frames 81 --fps 15 \
#     --guidance_scale 5.0 --seed 42 --dtype bfloat16 \
#     --enable_cpu_offload
# ──────────────────────────────────────────────────────────────────────────────
