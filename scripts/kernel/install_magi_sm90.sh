#!/usr/bin/env bash
#
# Install the minimal MagiAttention SM90 CUTLASS overlay used for H20
# validation. This requests the narrowest upstream-supported runtime matrix:
# BF16, head_dim=128, nfunc=1, forward, and backward. Expand the matrix only
# after this path is validated.
#
# Run this after `uv sync --extra gpu --dev`. A later exact `uv sync` removes
# the overlay, so rerun this script before the SM90 validation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
FFA_REV="ee1d15159cda6f3f97bfab9e487da146a8254970"
FFA_REQUIREMENT="ffa-fa3 @ git+https://github.com/demonatic/flash-attention.git@${FFA_REV}#subdirectory=hopper"
LOCAL_VERSION="sm90.bf16.hdim128.nfunc1.split32"

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

TORCH_CUDA_VERSION="$("${PYTHON}" - <<'PY'
import torch

print(torch.version.cuda or "")
PY
)"
if [[ -z "${TORCH_CUDA_VERSION}" ]]; then
  echo "The project PyTorch build does not provide CUDA support." >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  VERSIONED_CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VERSION}"
  if [[ -x "${VERSIONED_CUDA_HOME}/bin/nvcc" ]]; then
    CUDA_HOME="${VERSIONED_CUDA_HOME}"
  else
    CUDA_HOME="$("${PYTHON}" - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME

print(CUDA_HOME or "")
PY
)"
  fi
fi
if [[ -z "${CUDA_HOME}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA ${TORCH_CUDA_VERSION} toolkit is required. Set CUDA_HOME to a toolkit containing bin/nvcc." >&2
  exit 1
fi

NVCC_OUTPUT="$("${CUDA_HOME}/bin/nvcc" --version)"
if [[ "${NVCC_OUTPUT}" =~ release[[:space:]]([0-9]+\.[0-9]+) ]]; then
  NVCC_VERSION="${BASH_REMATCH[1]}"
else
  echo "Could not determine the CUDA version provided by ${CUDA_HOME}/bin/nvcc." >&2
  exit 1
fi
if [[ "${TORCH_CUDA_VERSION}" != "${NVCC_VERSION}" ]]; then
  echo "CUDA_HOME=${CUDA_HOME} provides CUDA ${NVCC_VERSION}, but PyTorch uses CUDA ${TORCH_CUDA_VERSION}." >&2
  echo "Set CUDA_HOME to a matching CUDA toolkit and rerun this script." >&2
  exit 1
fi

export CUDA_HOME
export CUDA_BIN_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -d "${CUDA_HOME}/compat" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

VISIBLE_ARCHS="$("${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("No visible CUDA device.")

architectures = sorted({torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())})
if architectures != [(9, 0)]:
    raise SystemExit(f"This minimal overlay targets SM90 only, found {architectures}.")

print(",".join(f"sm{major}{minor}" for major, minor in architectures))
PY
)"

validate_installed_overlay() {
  FFA_REV="${FFA_REV}" LOCAL_VERSION="${LOCAL_VERSION}" "${PYTHON}" - <<'PY'
import json
import os
from importlib.metadata import PackageNotFoundError, distribution

try:
    package = distribution("ffa-fa3")
except PackageNotFoundError:
    raise SystemExit(1)

if not package.version.endswith(f"+{os.environ['LOCAL_VERSION']}"):
    raise SystemExit(1)

direct_url_text = package.read_text("direct_url.json")
if direct_url_text is None:
    raise SystemExit(1)
vcs_info = json.loads(direct_url_text).get("vcs_info", {})
if vcs_info.get("commit_id") != os.environ["FFA_REV"]:
    raise SystemExit(1)

try:
    from flash_attn_cute.ffa_fa3.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
    from flash_attn_cute.ffa_fa3.flash_attn_config import CONFIG
except (ImportError, OSError):
    raise SystemExit(1)
if not callable(_flash_attn_forward) or not callable(_flash_attn_backward):
    raise SystemExit(1)

expected_flags = {
    "FLASHATTENTION_DISABLE_BACKWARD": False,
    "FLASHATTENTION_DISABLE_SPLIT": True,
    "FLASHATTENTION_DISABLE_PAGEDKV": True,
    "FLASHATTENTION_DISABLE_APPENDKV": True,
    "FLASHATTENTION_DISABLE_LOCAL": True,
    "FLASHATTENTION_DISABLE_SOFTCAP": True,
    "FLASHATTENTION_DISABLE_PACKGQA": True,
    "FLASHATTENTION_DISABLE_FP16": True,
    "FLASHATTENTION_DISABLE_FP8": True,
    "FLASHATTENTION_DISABLE_VARLEN": True,
    "FLASHATTENTION_DISABLE_CLUSTER": True,
    "FLASHATTENTION_DISABLE_HDIM64": True,
    "FLASHATTENTION_DISABLE_HDIM96": True,
    "FLASHATTENTION_DISABLE_HDIM128": False,
    "FLASHATTENTION_DISABLE_HDIM192": True,
    "FLASHATTENTION_DISABLE_HDIM256": True,
    "FLASHATTENTION_DISABLE_SM8x": True,
    "FLASHATTENTION_DISABLE_SM90": False,
    "FLASHATTENTION_ENABLE_VCOLMAJOR": False,
    "FLASH_ATTENTION_DISABLE_HDIMDIFF64": True,
    "FLASH_ATTENTION_DISABLE_HDIMDIFF192": True,
    "FLASHATTENTION_DISABLE_ARBITRARY": False,
    "FLASHATTENTION_NUM_FUNC": [1],
}
build_flags = CONFIG.get("build_flags", {})
if any(build_flags.get(name) != expected for name, expected in expected_flags.items()):
    raise SystemExit(1)
PY
}

if validate_installed_overlay; then
  echo "ffa-fa3 ${LOCAL_VERSION} from ${FFA_REV} is already installed and importable; nothing to do."
  exit 0
fi

# These flags constrain exposed dispatch and the standard kernels. The pinned
# upstream nfunc generator still instantiates some disabled dtype and feature
# combinations inside its nfunc translation units.
export FLASH_ATTENTION_DISABLE_APPENDKV="TRUE"
export FLASH_ATTENTION_DISABLE_ARBITRARY="FALSE"
export FLASH_ATTENTION_DISABLE_BACKWARD="FALSE"
export FLASH_ATTENTION_DISABLE_CLUSTER="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM128="FALSE"
export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM256="TRUE"
export FLASH_ATTENTION_DISABLE_HDIMDIFF64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIMDIFF192="TRUE"
export FLASH_ATTENTION_DISABLE_LOCAL="TRUE"
export FLASH_ATTENTION_DISABLE_PACKGQA="TRUE"
export FLASH_ATTENTION_DISABLE_PAGEDKV="TRUE"
export FLASH_ATTENTION_DISABLE_FP16="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
export FLASH_ATTENTION_DISABLE_SM80="TRUE"
export FLASH_ATTENTION_DISABLE_SM90="FALSE"
export FLASH_ATTENTION_ENABLE_VCOLMAJOR="FALSE"
export FLASH_ATTENTION_DISABLE_SOFTCAP="TRUE"
export FLASH_ATTENTION_DISABLE_SPLIT="TRUE"
export FLASH_ATTENTION_DISABLE_VARLEN="TRUE"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
# Match the pinned upstream ffa_fa3 Makefile default for this minimal
# validation. Evaluate its stable C++ API path with the full kernel matrix.
export FLASH_ATTENTION_FORCE_UNSTABLE_API="TRUE"
export FLASH_ATTENTION_NUM_FUNC="1"
export FLASH_ATTN_LOCAL_VERSION="${LOCAL_VERSION}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+${NVCC_APPEND_FLAGS} }--split-compile=32"

echo "Installing MagiAttention CUTLASS FFA for ${VISIBLE_ARCHS} with CUDA ${NVCC_VERSION}."
echo "Requested runtime matrix: BF16, head_dim=128, nfunc=1, forward+backward, split-compile=32."
uv pip install \
  --python "${PYTHON}" \
  --no-build-isolation \
  --no-deps \
  --no-cache \
  --reinstall \
  "${FFA_REQUIREMENT}"

if ! validate_installed_overlay; then
  echo "Installed MagiAttention SM90 CUTLASS FFA backend failed provenance or import validation." >&2
  exit 1
fi
echo "MagiAttention SM90 CUTLASS FFA backend provenance and import validation succeeded."

echo "If a later 'uv sync' removes this overlay, rerun this script before SM90 validation."
