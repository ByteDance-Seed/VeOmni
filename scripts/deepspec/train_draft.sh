#!/bin/bash
# Launch DeepSpec draft-model training on VeOmni.
#
# Usage:
#   bash scripts/deepspec/train_draft.sh configs/deepspec/dspark_qwen3_4b.yaml [extra overrides...]
#
# Env:
#   DEEPSPEC_PATH   Path to the DeepSpec repo root (defaults to a sibling
#                   ``DeepSpec/`` checkout next to this VeOmni repo).
#   CUDA_VISIBLE_DEVICES / NPROC_PER_NODE / NNODES / NODE_RANK / MASTER_* — as
#   understood by the top-level train.sh (torchrun) launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VEOMNI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${VEOMNI_ROOT}/.." && pwd)"

# Default DEEPSPEC_PATH to a sibling checkout if unset.
if [[ -z "${DEEPSPEC_PATH:-}" && -d "${WORKSPACE_ROOT}/DeepSpec/deepspec" ]]; then
  export DEEPSPEC_PATH="${WORKSPACE_ROOT}/DeepSpec"
fi
echo "[train_draft] DEEPSPEC_PATH=${DEEPSPEC_PATH:-<unset>}"

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/deepspec/train_draft.sh <config.yaml> [overrides...]" >&2
  exit 1
fi

CONFIG="$1"; shift

cd "${VEOMNI_ROOT}"
bash train.sh tasks/train_deepspec_draft.py "${CONFIG}" "$@"
