#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "$0")" || exit 1

OUTPUT_ROOT="${JANUS_V2_OUTPUT_ROOT:-outputs/janus_v2}"
LOG_DIR="${JANUS_V2_LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_train_$(date +%Y%m%d_%H%M%S).log"

{
bash scripts/seed_omni/debug_omni.sh mixed || exit "$?"
bash scripts/seed_omni/debug_omni.sh understanding || exit "$?"
bash scripts/seed_omni/debug_omni.sh t2i || exit "$?"
} 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
