#!/bin/bash
# One-click smoke test for the SeedOmni V2 (omni_v2) models.
#
# For each selected model it runs a SHORT training run (a couple of optimizer
# steps) followed by a SHORT inference run, then prints a PASS/FAIL summary.
# The goal is a fast "does the code still run end to end" regression check after
# refactoring -- NOT a convergence or quality test.
#
# Usage:
#   bash scripts/seed_omni/run_smoke.sh                 # run all models
#   bash scripts/seed_omni/run_smoke.sh qwen3 janus     # run a subset
#   STEPS=1 bash scripts/seed_omni/run_smoke.sh qwen3   # 1 training step only
#   SKIP_TRAIN=1 bash scripts/seed_omni/run_smoke.sh    # inference only
#   SKIP_INFER=1 bash scripts/seed_omni/run_smoke.sh    # training only
#
# Models (case-insensitive):
#   janus      Janus-1.3B          (train + FSDP/emb-parallel inference)
#   qwen3      Qwen3-0.6B          (train + eager text inference)
#   qwen3_it   Qwen3-0.6B vis IT   (train + eager image-understanding inference)
#   qwen3vl    Qwen3-VL-2B         (train + eager image-understanding inference)
#   qwen3moe   Qwen3-30B-A3B (MoE) (train + FSDP/EP inference)
#
# Env knobs:
#   STEPS            training steps per model            (default 2)
#   MAX_NEW_TOKENS   generated tokens per inference      (default 16)
#   OUT              scratch output root                 (default /tmp/seed_omni_smoke)
#   SKIP_TRAIN=1     skip the training phase
#   SKIP_INFER=1     skip the inference phase
#
# Notes:
#   * Checkpoints are assumed to already exist at the paths in each base.yaml.
#     A missing split checkpoint surfaces as a FAIL for that model.
#   * Per-model logs land in $OUT/<model>/{train.log,infer.log}.
#   * janus inference uses the distributed (FSDP2 + emb-parallel) module set, and
#     qwen3moe inference uses the distributed (FSDP2 + EP) module set -- both
#     launched via torchrun. The other models run eager single-process inference.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Run inside your already-activated training environment (the one with torch +
# torchrun, e.g. the `veomni` env). Fail fast with a clear hint otherwise.
if ! command -v torchrun >/dev/null 2>&1 || ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: 'torchrun' / 'torch' not found on PATH." >&2
  echo "Activate your training environment first (e.g. the 'veomni' env), then re-run." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-offline}

STEPS=${STEPS:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
OUT=${OUT:-/tmp/seed_omni_smoke}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_INFER=${SKIP_INFER:-0}

# qwen3_it inference loads a TRAINED split checkpoint: the base split's ViT
# merger is retargeted (out_hidden_size 1024) + reinit during training, so the
# raw base split cannot be inferred directly. Point this at the assembled
# instruction-tuned checkpoint (same split format as the other models).
QWEN3_IT_CKPT=${QWEN3_IT_CKPT:-/mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B-Visual-Instruction-Tuning}
# Image used by understanding inference. Falls back to a generated gray PNG if
# the path is missing.
IMAGE=${IMAGE:-/mnt/hdfs/user_dir/veomni_omni/models/transformers/Janus-1.3B/teaser.png}
# qwen3_it uses its own understanding image (seedream debug sample).
QWEN3_IT_IMAGE=${QWEN3_IT_IMAGE:-/mnt/hdfs/user_dir/seedream/debug/output/sample_1/0.png}

CFG_DIR="configs/seed_omni"
TRAIN_PY="tasks/omni/train_omni.py"
INFER_PY="tasks/omni/infer_omni.py"

ALL_MODELS=(janus qwen3 qwen3_it qwen3vl qwen3moe)

# Selected models (default: all). Normalize to lower-case.
if [[ $# -gt 0 ]]; then
  MODELS=()
  for m in "$@"; do MODELS+=("$(echo "$m" | tr '[:upper:]' '[:lower:]')"); done
else
  MODELS=("${ALL_MODELS[@]}")
fi

mkdir -p "$OUT"

# Image for the image-understanding inference paths: use $IMAGE if present, else
# generate a flat-gray fallback. Resolves $TEST_IMAGE (consumed by the cases).
TEST_IMAGE="$IMAGE"
make_test_image() {
  [[ -f "$TEST_IMAGE" ]] && return 0
  TEST_IMAGE="$OUT/assets/test_image.png"
  [[ -f "$TEST_IMAGE" ]] && return 0
  mkdir -p "$(dirname "$TEST_IMAGE")"
  python - "$TEST_IMAGE" <<'PY'
import sys
from PIL import Image
Image.new("RGB", (256, 256), (127, 127, 127)).save(sys.argv[1])
print("wrote test image:", sys.argv[1])
PY
}

# Common training overrides: cap steps, disable wandb + checkpoint writes.
train_overrides() {
  local outdir="$1"
  echo "--train.max_steps $STEPS \
        --train.num_train_epochs 1 \
        --train.wandb.enable false \
        --train.checkpoint.output_dir $outdir \
        --train.checkpoint.save_steps 100000000 \
        --train.checkpoint.save_epochs 0 \
        --train.checkpoint.hf_save_steps 100000000 \
        --train.checkpoint.hf_save_epochs 0"
}

# Common inference overrides: short generation, scratch output dir.
infer_overrides() {
  local outdir="$1"
  echo "--infer.output_dir $outdir \
        --infer.generation_kwargs.max_new_tokens $MAX_NEW_TOKENS"
}

declare -A CONFIG=(
  [janus]="$CFG_DIR/Janus/janus_1.3b/base.yaml"
  [qwen3]="$CFG_DIR/Qwen/qwen3_0.6b/base.yaml"
  [qwen3_it]="$CFG_DIR/Qwen/qwen3_0.6b/visual_instruction_tuning.yaml"
  [qwen3vl]="$CFG_DIR/Qwen/qwen3vl_2b/base.yaml"
  [qwen3moe]="$CFG_DIR/Qwen/qwen3_30b_a3b/base.yaml"
)

declare -A TRAIN_STATUS
declare -A INFER_STATUS

# Run one command, tee to a log, return the command's (not tee's) exit code.
run_logged() {
  local log="$1"; shift
  echo "+ $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

run_train() {
  local name="$1" outdir="$2" log="$3"
  # shellcheck disable=SC2046
  run_logged "$log" bash train.sh "$TRAIN_PY" "${CONFIG[$name]}" $(train_overrides "$outdir/ckpt")
}

# Inference is model-specific: scenario, eager vs distributed, image vs text.
run_infer() {
  local name="$1" outdir="$2" log="$3"
  local common; common=$(infer_overrides "$outdir/infer")
  case "$name" in
    janus)
      # Distributed (FSDP2 + emb-parallel) text-to-image generation.
      # T2I needs the full image-token grid, so override the shared short cap.
      # shellcheck disable=SC2046
      run_logged "$log" bash train.sh "$INFER_PY" "${CONFIG[$name]}" \
        --model.model_config.modules "$CFG_DIR/Janus/janus_1.3b/modules_infer_fsdp.yaml" \
        --model.model_config.infer_type infer_gen \
        --infer.prompt "A photo of a cat sitting on a chair." \
        $common \
        --infer.generation_kwargs.max_new_tokens 2048
      ;;
    qwen3)
      # shellcheck disable=SC2046
      run_logged "$log" python "$INFER_PY" "${CONFIG[$name]}" \
        --infer.prompt "What is 2+2? Answer briefly." \
        $common
      ;;
    qwen3_it)
      # Inference loads the TRAINED checkpoint ($QWEN3_IT_CKPT). The text encoder
      # / llm rebase under --model.model_path; the ViT module path is absolute in
      # the YAML, so override it to the trained vision sub-checkpoint.
      local it_image="$QWEN3_IT_IMAGE"
      [[ -f "$it_image" ]] || it_image="$TEST_IMAGE"
      # shellcheck disable=SC2046
      run_logged "$log" python "$INFER_PY" "${CONFIG[$name]}" \
        --model.model_path "$QWEN3_IT_CKPT" \
        --model.model_config.modules.qwen3vl_vision.model_path "$QWEN3_IT_CKPT/qwen3vl_vision" \
        --infer.prompt "Describe this image briefly." \
        --infer.image "$it_image" \
        $common
      ;;
    qwen3vl)
      # shellcheck disable=SC2046
      run_logged "$log" python "$INFER_PY" "${CONFIG[$name]}" \
        --infer.prompt "Describe this image briefly." \
        --infer.image "$TEST_IMAGE" \
        $common
      ;;
    qwen3moe)
      # Distributed (FSDP2 + Expert-Parallel) text inference (eager would OOM at 30B).
      # shellcheck disable=SC2046
      run_logged "$log" bash train.sh "$INFER_PY" "${CONFIG[$name]}" \
        --model.model_config.modules "$CFG_DIR/Qwen/qwen3_30b_a3b/modules_infer_fsdp.yaml" \
        --infer.prompt "What is 2+2? Answer briefly." \
        $common
      ;;
    *)
      echo "unknown model: $name" | tee "$log"
      return 2
      ;;
  esac
}

needs_image() {
  case "$1" in qwen3_it|qwen3vl) return 0;; *) return 1;; esac
}

for name in "${MODELS[@]}"; do
  if [[ -z "${CONFIG[$name]:-}" ]]; then
    echo "!! skipping unknown model '$name' (valid: ${ALL_MODELS[*]})"
    TRAIN_STATUS[$name]="UNKNOWN"
    INFER_STATUS[$name]="UNKNOWN"
    continue
  fi

  outdir="$OUT/$name"
  mkdir -p "$outdir"
  echo
  echo "==================== $name ===================="
  echo "config: ${CONFIG[$name]}"

  if [[ "$SKIP_TRAIN" == "1" ]]; then
    TRAIN_STATUS[$name]="SKIP"
  else
    echo "---- train ($STEPS steps) ----"
    if run_train "$name" "$outdir" "$outdir/train.log"; then
      TRAIN_STATUS[$name]="PASS"
    else
      TRAIN_STATUS[$name]="FAIL"
    fi
  fi

  if [[ "$SKIP_INFER" == "1" ]]; then
    INFER_STATUS[$name]="SKIP"
  else
    needs_image "$name" && make_test_image
    echo "---- infer ----"
    if run_infer "$name" "$outdir" "$outdir/infer.log"; then
      INFER_STATUS[$name]="PASS"
    else
      INFER_STATUS[$name]="FAIL"
    fi
  fi
done

echo
echo "==================== summary ===================="
printf "%-12s %-8s %-8s\n" "model" "train" "infer"
printf "%-12s %-8s %-8s\n" "-----" "-----" "-----"
fail=0
for name in "${MODELS[@]}"; do
  t="${TRAIN_STATUS[$name]:-N/A}"
  i="${INFER_STATUS[$name]:-N/A}"
  printf "%-12s %-8s %-8s\n" "$name" "$t" "$i"
  [[ "$t" == "FAIL" || "$i" == "FAIL" || "$t" == "UNKNOWN" ]] && fail=1
done
echo "logs under: $OUT/<model>/{train.log,infer.log}"
echo "================================================="

exit "$fail"
