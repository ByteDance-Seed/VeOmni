#!/bin/bash
# Step through Janus training one graph node at a time — inspect conversation_list
# in ipdb (rank 0 only).
#
# Breakpoints live in each Janus module entry (siglip / vqvae / text_encoder /
# llama) via conversation.ipdb_conversation_break().  They fire when
# VEOMNI_DEBUG_CONV_IPDB=1.
#
# Usage:
#   bash scripts/seed_omni/debug_conversation_ipdb.sh [understanding|t2i|mixed|all]
#
# Dataset modes (see make_janus_omni_demo.py --dataset_mode):
#   understanding — pure I2T (user image + question → assistant text)
#   t2i           — pure T2I (user prompt → assistant image)
#   mixed         — interleave UG (user text+image+text, assistant image+text)
#   all           — all three row kinds (default: mixed for compact ipdb)
#
# Override with env: DATASET_MODE=t2i bash scripts/seed_omni/debug_conversation_ipdb.sh
#
# ipdb cheatsheet (rank 0 terminal):
#   conversation_list          # raw batch carrier
#   conversation_list[0]       # first sample
#   conversation_list[0][0]    # first item → .type / .value / .meta
#   [p.type for p in conversation_list[0]]
#   [p.meta.get('role') for p in conversation_list[0]]
#   n / s / c                  # next / step / continue to next breakpoint
#
# Rank behaviour:
#   - rank 0: enters ipdb at each breakpoint (prints summarize_conversation_batch)
#   - rank 1: waits on dist.barrier() (no ipdb prompt) until rank 0 continues
#
# Requirements:
#   - 2 GPUs (FSDP2 meta-init needs >1 rank in this repo)
#   - TTY attached (do NOT pipe through tee — breaks ipdb readline)
#   - pip install ipdb  (usually already in dev extras)
#
# Expected breakpoint order (one training step, Janus train.yaml):
#   janus_siglip.pre_forward → janus_siglip.post_forward
#   janus_vqvae.pre_forward (encode) → janus_vqvae.encode → janus_vqvae.post_forward
#   janus_text_encoder.pre_forward → janus_text_encoder.encode
#   janus_llama.pre_forward → janus_llama.forward
#   janus_text_encoder.decode
#   janus_vqvae.pre_forward (decode) → janus_vqvae.decode

set -o pipefail

DATASET_MODE="${DATASET_MODE:-}"
if [[ $# -gt 0 && "$1" =~ ^(understanding|t2i|mixed|all)$ ]]; then
  DATASET_MODE="$1"
  shift
fi
DATASET_MODE="${DATASET_MODE:-mixed}"

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

# Activate a venv only when none is active.  Do not override the user's
# ``(veomni)`` shell — a bare repo ``.venv`` may lack pyarrow / gpu extras.
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
    echo "[debug_conversation_ipdb] warning: no venv found; using PATH python: $(command -v python || echo missing)" >&2
  fi
fi

if ! python -c "import pyarrow" 2>/dev/null; then
  echo "[debug_conversation_ipdb] ERROR: pyarrow is not installed in the active Python:" >&2
  python -c "import sys; print('  ', sys.executable)" 2>/dev/null || true
  echo "[debug_conversation_ipdb] Install into this venv, e.g.:" >&2
  echo "  uv pip install pyarrow" >&2
  echo "Or activate the full VeOmni env first:" >&2
  echo "  source /app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate" >&2
  exit 1
fi

unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      NPROC_PER_NODE NNODES MASTER_ADDR MASTER_PORT \
      PET_NPROC_PER_NODE PET_NNODES PET_MASTER_ADDR PET_MASTER_PORT \
      TORCHELASTIC_RUN_ID

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

# Enable conversation-list ipdb breakpoints (rank 0 only).
export VEOMNI_DEBUG_CONV_IPDB=1

DEBUG_DATA="${REPO_ROOT}/outputs/janus_conversation_ipdb_debug/data"
mkdir -p "${DEBUG_DATA}"

echo "[debug_conversation_ipdb] building demo parquet (dataset_mode=${DATASET_MODE}) ..."
python "${REPO_ROOT}/scripts/multimodal/convert_data/make_janus_omni_demo.py" \
  --dataset_mode "${DATASET_MODE}" \
  --out_dir "${DEBUG_DATA}" \
  --num_repeat 4 \
  --und_reply "${REPO_ROOT}/janus_out/infer_und/reply.txt" \
  --gen_image "${REPO_ROOT}/janus_out/infer_gen/generated_image_0.png"

PARQUET="${DEBUG_DATA}/janus_omni_demo_${DATASET_MODE}.parquet"
if [[ ! -f "${PARQUET}" ]]; then
  echo "[debug_conversation_ipdb] ERROR: parquet not found at ${PARQUET}" >&2
  exit 1
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  tasks/train_omni.py configs/seed_omni/janus_1.3b/veomni_janus.yaml \
  --train.global_batch_size 2 \
  --train.micro_batch_size 1 \
  --train.max_steps 1 \
  --train.num_train_epochs 1 \
  --train.gradient_checkpointing.enable false \
  --train.wandb.enable false \
  --train.checkpoint.save_steps 100000000 \
  --train.checkpoint.save_epochs 0 \
  --train.checkpoint.hf_save_steps 100000000 \
  --train.checkpoint.save_hf_weights false \
  --train.checkpoint.output_dir "outputs/janus_conversation_ipdb_debug/${DATASET_MODE}" \
  --data.train_path "${PARQUET}" \
  --data.dataloader.num_workers 0 \
  "$@"
