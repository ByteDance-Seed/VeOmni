#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "$0")" || exit 1

OUTPUT_ROOT="${JANUS_V2_OUTPUT_ROOT:-outputs/janus_v2}"
JANUS_OUT="${JANUS_V2_JANUS_OUT:-${OUTPUT_ROOT}/janus_out}"
LOG_DIR="${JANUS_V2_LOG_DIR:-${OUTPUT_ROOT}/logs}"
MODEL_HUB="${MODEL_HUB:-/mnt/hdfs/user_dir/veomni_omni/models}"
mkdir -p "${JANUS_OUT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_gen_$(date +%Y%m%d_%H%M%S).log"

{
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
  if _pick_venv "$(pwd)/.venv/bin/activate"; then
    :
  elif _pick_venv "/app/VeOmni/submodules/Open-VeOmni/.venv/bin/activate"; then
    :
  else
    echo "[test_gen] warning: no venv found; using PATH python: $(command -v python || echo missing)" >&2
  fi
fi
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python tasks/omni/infer_omni.py \
    configs/seed_omni/janus_1.3b/veomni_janus.yaml \
    --model.omni_infer_type infer_gen \
    --infer.model_path "${MODEL_HUB}/seed_omni/Janus-1.3B" \
    --infer.prompt "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue." \
    --infer.output_dir "${JANUS_OUT}" \
    --infer.generation_kwargs.max_new_tokens 2048 \
    --infer.generation_kwargs.guidance_scale 5.0
} 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
