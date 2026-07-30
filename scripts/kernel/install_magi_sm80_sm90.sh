#!/usr/bin/env bash
#
# Install MagiAttention's architecture-specific BF16 CUTLASS FFA backend.
#
# Run this after `uv sync --extra gpu --dev`. A later exact `uv sync` removes
# this overlay, so rerun the script before using MagiAttention on SM80/SM90.
#
# Examples:
#   bash scripts/kernel/install_magi_sm80_sm90.sh sm80
#   bash scripts/kernel/install_magi_sm80_sm90.sh sm90
#   bash scripts/kernel/install_magi_sm80_sm90.sh sm80,sm90
#   bash scripts/kernel/install_magi_sm80_sm90.sh auto

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
FFA_REV="ee1d15159cda6f3f97bfab9e487da146a8254970"
FFA_REQUIREMENT="ffa-fa3 @ git+https://github.com/demonatic/flash-attention.git@${FFA_REV}#subdirectory=hopper"

usage() {
  echo "Usage: $0 [auto|sm80|sm90|sm80,sm90]" >&2
}

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}. Run 'uv sync --extra gpu --dev' first." >&2
  exit 1
fi

for package in torch flash_attn_cute; do
  if ! "${PYTHON}" -c "import ${package}" >/dev/null 2>&1; then
    echo "Missing ${package}. Run 'uv sync --extra gpu --dev' before this script." >&2
    exit 1
  fi
done

ARCHS="${1:-auto}"
if [[ "${ARCHS}" == "auto" ]]; then
  ARCHS="$("${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("No visible CUDA device; pass sm80 or sm90 explicitly on a GPU host.")

architectures = sorted({torch.cuda.get_device_capability(index)[0] for index in range(torch.cuda.device_count())})
unsupported = [major for major in architectures if major not in (8, 9)]
if unsupported:
    if all(major >= 10 for major in unsupported) and len(unsupported) == len(architectures):
        print("cute")
        raise SystemExit(0)
    raise SystemExit(f"Auto-detection found unsupported mixed CUDA capabilities: {architectures}")

print(",".join(f"sm{major}0" for major in architectures))
PY
  )"
fi

case "${ARCHS}" in
  cute)
    echo "SM100+ uses the CUTE DSL backend installed by 'uv sync --extra gpu --dev'; no overlay is needed."
    exit 0
    ;;
  sm80)
    DISABLE_SM80="FALSE"
    DISABLE_SM90="TRUE"
    LOCAL_VERSION="sm80.bf16.nfunc13"
    ;;
  sm90)
    DISABLE_SM80="TRUE"
    DISABLE_SM90="FALSE"
    LOCAL_VERSION="sm90.bf16.nfunc13"
    ;;
  sm80,sm90 | sm90,sm80)
    DISABLE_SM80="FALSE"
    DISABLE_SM90="FALSE"
    LOCAL_VERSION="sm80sm90.bf16.nfunc13"
    ;;
  *)
    usage
    echo "SM100+ does not use this script; its CUTE backend is part of the gpu extra." >&2
    exit 2
    ;;
esac

INSTALLED_VERSION="$("${PYTHON}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("ffa-fa3"))
except PackageNotFoundError:
    pass
PY
)"
if [[ "${INSTALLED_VERSION}" == *"+${LOCAL_VERSION}" ]]; then
  echo "ffa-fa3 ${INSTALLED_VERSION} is already installed; nothing to do."
  exit 0
fi

CUDA_HOME="${CUDA_HOME:-$("${PYTHON}" - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME

print(CUDA_HOME or "")
PY
)}"
if [[ -z "${CUDA_HOME}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "A CUDA toolkit matching PyTorch is required. Set CUDA_HOME to a toolkit containing bin/nvcc." >&2
  exit 1
fi

TORCH_CUDA_VERSION="$("${PYTHON}" - <<'PY'
import torch

print(torch.version.cuda or "")
PY
)"
NVCC_VERSION="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "${TORCH_CUDA_VERSION}" || -z "${NVCC_VERSION}" ]]; then
  echo "Could not determine the PyTorch or nvcc CUDA version." >&2
  exit 1
fi
if [[ "${TORCH_CUDA_VERSION%%.*}" != "${NVCC_VERSION%%.*}" ]]; then
  echo "CUDA_HOME=${CUDA_HOME} provides CUDA ${NVCC_VERSION}, but PyTorch uses CUDA ${TORCH_CUDA_VERSION}." >&2
  echo "Set CUDA_HOME to a matching CUDA toolkit and rerun this script." >&2
  exit 1
fi

export CUDA_HOME
export CUDA_BIN_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export FLASH_ATTENTION_DISABLE_APPENDKV="TRUE"
export FLASH_ATTENTION_DISABLE_CLUSTER="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM256="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM64="FALSE"
export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
export FLASH_ATTENTION_DISABLE_HDIMDIFF192="TRUE"
export FLASH_ATTENTION_DISABLE_HDIMDIFF64="TRUE"
export FLASH_ATTENTION_DISABLE_LOCAL="TRUE"
export FLASH_ATTENTION_DISABLE_PACKGQA="TRUE"
export FLASH_ATTENTION_DISABLE_PAGEDKV="TRUE"
export FLASH_ATTENTION_DISABLE_FP16="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
export FLASH_ATTENTION_DISABLE_SM80="${DISABLE_SM80}"
export FLASH_ATTENTION_DISABLE_SM90="${DISABLE_SM90}"
export FLASH_ATTENTION_DISABLE_SOFTCAP="TRUE"
export FLASH_ATTENTION_DISABLE_SPLIT="TRUE"
export FLASH_ATTENTION_DISABLE_VARLEN="TRUE"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export FLASH_ATTENTION_FORCE_UNSTABLE_API="TRUE"
export FLASH_ATTENTION_NUM_FUNC="1,3"
export FLASH_ATTN_LOCAL_VERSION="${LOCAL_VERSION}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+${NVCC_APPEND_FLAGS} }--split-compile=${NVCC_THREADS}"

echo "Installing MagiAttention BF16 CUTLASS FFA for ${ARCHS} (head_dim=64,128; nfunc=1,3) with CUDA ${NVCC_VERSION}."
uv pip install \
  --python "${PYTHON}" \
  --no-build-isolation \
  --no-deps \
  --no-cache \
  --reinstall \
  "${FFA_REQUIREMENT}"

"${PYTHON}" - <<'PY'
from flash_attn_cute.ffa_fa3.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

assert _flash_attn_forward is not None
assert _flash_attn_backward is not None
print("MagiAttention CUTLASS FFA backend import succeeded.")
PY

echo "If a later 'uv sync' removes this overlay, rerun this script before using magi_attention."
