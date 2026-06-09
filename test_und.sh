#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "$0")" || exit 1

OUTPUT_ROOT="${JANUS_V2_OUTPUT_ROOT:-outputs/janus_v2}"
JANUS_OUT="${JANUS_V2_JANUS_OUT:-${OUTPUT_ROOT}/janus_out}"
LOG_DIR="${JANUS_V2_LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${JANUS_OUT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_und_$(date +%Y%m%d_%H%M%S).log"

{
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python tasks/omni/infer_omni.py \
    configs/seed_omni/janus_1.3b/veomni_janus.yaml \
    --model.omni_infer_type infer_und \
    --infer.model_path /mnt/hdfs/shizhelun/veomni_omni/models/seed_omni/Janus-1.3B \
    --infer.prompt "What do you see in this image?" \
    --infer.image /mnt/hdfs/shizhelun/veomni_omni/models/transformers/Janus-1.3B/teaser.png \
    --infer.output_dir "${JANUS_OUT}" \
    --infer.generation_kwargs.max_new_tokens 1024
} 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
