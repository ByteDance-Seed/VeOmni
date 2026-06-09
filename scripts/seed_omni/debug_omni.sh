#!/bin/bash
# Janus OmniTrainer smoke run — fixed 8 GPUs, default 20 optimizer steps.
#
# Usage:
#   bash scripts/seed_omni/debug_omni.sh [understanding|t2i|mixed|all]
#
# Dataset modes (make_janus_omni_demo.py --dataset_mode):
#   understanding | t2i | mixed (default) | all

set -o pipefail

readonly NUM_GPUS=8

DATASET_MODE="${DATASET_MODE:-}"
if [[ $# -gt 0 && "$1" =~ ^(understanding|t2i|mixed|all)$ ]]; then
  DATASET_MODE="$1"
  shift
fi
DATASET_MODE="${DATASET_MODE:-mixed}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-20}"

# train_steps ≈ floor(len(dataset) / global_batch_size); need enough parquet rows.
NUM_REPEAT="${NUM_REPEAT:-$((MAX_STEPS * GLOBAL_BATCH_SIZE))}"

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  _pick_venv() {
    local activate_path="$1"
    if [[ -f "${activate_path}" ]]; then
      # shellcheck source=/dev/null
      source "${activate_path}"
      return 0
    fi
    return 1
  }
  if _pick_venv "${REPO_ROOT}/.venv/bin/activate"; then
    :
  elif _pick_venv "/app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate"; then
    :
  else
    echo "[debug_omni] warning: no venv found; using PATH python: $(command -v python || echo missing)" >&2
  fi
fi

if ! python -c "import pyarrow" 2>/dev/null; then
  echo "[debug_omni] ERROR: pyarrow is not installed in the active Python:" >&2
  python -c "import sys; print('  ', sys.executable)" 2>/dev/null || true
  echo "[debug_omni] Install into this venv, e.g.:" >&2
  echo "  cd ${REPO_ROOT} && uv sync --extra gpu --extra dit --group dev" >&2
  exit 1
fi

unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      NPROC_PER_NODE NNODES MASTER_ADDR MASTER_PORT \
      PET_NPROC_PER_NODE PET_NNODES PET_MASTER_ADDR PET_MASTER_PORT \
      TORCHELASTIC_RUN_ID

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

OUTPUT_ROOT="${JANUS_V2_OUTPUT_ROOT:-${REPO_ROOT}/outputs/janus_v2}"
JANUS_OUT="${JANUS_V2_JANUS_OUT:-${OUTPUT_ROOT}/janus_out}"
DEBUG_DATA="${JANUS_V2_DATA_DIR:-${OUTPUT_ROOT}/data}"
TRAIN_DEBUG="${JANUS_V2_TRAIN_DEBUG_DIR:-${OUTPUT_ROOT}/train_debug}"
MODEL_HUB="${MODEL_HUB:-/mnt/hdfs/user_dir/veomni_omni/models}"
mkdir -p "${DEBUG_DATA}"

echo "[debug_omni] building demo parquet (mode=${DATASET_MODE}, num_repeat=${NUM_REPEAT}) ..."
python "${REPO_ROOT}/scripts/multimodal/convert_data/make_janus_omni_demo.py" \
  --dataset_mode "${DATASET_MODE}" \
  --out_dir "${DEBUG_DATA}" \
  --num_repeat "${NUM_REPEAT}" \
  --und_reply "${JANUS_OUT}/infer_und/reply.txt" \
  --gen_image "${JANUS_OUT}/infer_gen/generated_image_0.png"

PARQUET="${DEBUG_DATA}/janus_omni_demo_${DATASET_MODE}.parquet"
if [[ ! -f "${PARQUET}" ]]; then
  echo "[debug_omni] ERROR: parquet not found at ${PARQUET}" >&2
  exit 1
fi

echo "[debug_omni] ${NUM_GPUS} GPUs, global_batch_size=${GLOBAL_BATCH_SIZE}, max_steps=${MAX_STEPS}"
echo "[debug_omni] expected train_steps ≈ min(max_steps, floor(${NUM_REPEAT}/${GLOBAL_BATCH_SIZE}))"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  tasks/omni/train_omni.py configs/seed_omni/janus_1.3b/veomni_janus.yaml \
  --model.model_path "${MODEL_HUB}/seed_omni/Janus-1.3B" \
  --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
  --train.max_steps "${MAX_STEPS}" \
  --train.num_train_epochs 1 \
  --train.gradient_checkpointing.enable false \
  --train.wandb.enable false \
  --train.checkpoint.save_steps 100000000 \
  --train.checkpoint.save_epochs 0 \
  --train.checkpoint.hf_save_steps 100000000 \
  --train.checkpoint.save_hf_weights false \
  --train.checkpoint.output_dir "${TRAIN_DEBUG}/${DATASET_MODE}" \
  --data.train_path "${PARQUET}" \
  --data.dataloader.num_workers 0 \
  --data.dataloader.drop_last true \
  "$@"
