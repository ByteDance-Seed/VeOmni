#!/bin/bash
# Minimal 2-GPU launcher for stepping through OmniModel V2 training with ipdb.
# (FSDP2 + sharded meta-init needs a real >1-rank device mesh, so this runs 2 ranks.)
#
# Why this exists (vs. train.sh):
#   - train.sh pipes through `| tee log.txt`, so stdout is NOT a TTY and
#     interactive ipdb/pdb prompts break (no readline, no prompt flush).
#   - This script runs 2 ranks (nproc=2, required for FSDP2 meta-init) with
#     stdio inherited from your terminal, so `breakpoint()` / `ipdb.set_trace()`
#     attach. See the rank-guard note below to debug a single process cleanly.
#
# Usage:
#   1. Drop a breakpoint anywhere in the code you want to inspect, e.g.
#        breakpoint()                      # uses ipdb via PYTHONBREAKPOINT below
#      or explicitly:
#        import ipdb; ipdb.set_trace()
#   2. Run:
#        bash scripts/seed_omni/debug_omni_2gpu.sh
#   3. Step with ipdb (n / s / c / p <expr> / w / l ...).
#
# Notes:
#   - DP=2 (two ranks): FSDP2 + meta-init shards parameters across a real device
#     mesh, so a single rank can't materialise the model — we need >1 GPU.
#   - micro_batch_size=1, global_batch_size=2  → with dp_size=2 that's exactly
#     ONE micro-batch per rank, grad-accum = 1. Easiest flow to follow.
#   - ipdb with 2 ranks: both processes share this terminal, so prompts/stdout
#     from rank 0 and rank 1 interleave. Guard breakpoints by rank to debug one
#     process cleanly, e.g.:
#         import torch.distributed as dist
#         if not dist.is_initialized() or dist.get_rank() == 0:
#             breakpoint()
#   - gradient_checkpointing is DISABLED — GC re-runs the forward under
#     torch.utils.checkpoint, which double-executes module code and makes
#     stepping very confusing. Re-enable only when you specifically debug GC.
#   - dataloader.num_workers=1: the seedomni transform runs in a *worker*
#     process, so breakpoints inside the transform will NOT attach. Module
#     forward/pre_forward/encode/decode run in THIS (main) process, so model
#     breakpoints work. To also step the transform, set num_workers=0 — but
#     note PyTorch forbids prefetch_factor with 0 workers, so you'd also need
#     to patch veomni/data/data_loader.py to drop prefetch_factor in that case.

set -o pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd)"

# --- environment -----------------------------------------------------------
# Activate venv (try the in-repo .venv first, then the known /app path).
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/.venv/bin/activate"
elif [[ -f /app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate ]]; then
  source /app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate
fi

# Clear any torch-elastic / rank env inherited from a previous multi-proc run.
# Otherwise torchrun (or the arg parser) can pick up a stale WORLD_SIZE / nproc
# and spawn rank 1 onto cuda:1 — which fails with "invalid device ordinal" when
# only one GPU is exposed below.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      NPROC_PER_NODE NNODES MASTER_ADDR MASTER_PORT \
      PET_NPROC_PER_NODE PET_NNODES PET_MASTER_ADDR PET_MASTER_PORT \
      TORCHELASTIC_RUN_ID

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
# Pin to two valid GPUs (FSDP2 needs >1 for the sharded meta-init). Override with
# e.g. `CUDA_VISIBLE_DEVICES=2,3 bash ...`; keep it to TWO ids to match nproc=2.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
# Make bare `breakpoint()` calls open ipdb instead of pdb.
export PYTHONBREAKPOINT=${PYTHONBREAKPOINT:-ipdb.set_trace}

# --- launch (2 ranks for FSDP2, stdio attached → ipdb works) ---------------
# No `| tee`, no `> log.txt` redirect: keep stdout/stdin as the terminal.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  tasks/train_omni.py configs/seed_omni/janus_1.3b/veomni_janus.yaml \
  --train.global_batch_size 2 \
  --train.micro_batch_size 1 \
  --train.num_train_epochs 1 \
  --train.gradient_checkpointing.enable false \
  --train.wandb.enable false \
  --train.checkpoint.save_steps 100000000 \
  --train.checkpoint.save_epochs 0 \
  --train.checkpoint.hf_save_steps 100000000 \
  --train.checkpoint.save_hf_weights false \
  --train.checkpoint.output_dir outputs/janus_omni_debug \
  --data.dataloader.num_workers 1 \
  "$@"
