#!/usr/bin/env bash
#
# Install the MagiAttention SM90 CUTLASS overlay used by VeOmni. The verified
# default matrix covers BF16/FP16, input head dimensions up to 128 through the
# hdim128 bucket, and arbitrary masks requiring nfunc 1, 3, or 5. Command-line
# options can request a different upstream build matrix.
#
# Run this after `uv sync --extra gpu --dev`. A later exact `uv sync` removes
# the overlay, so rerun this script before the SM90 validation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
FFA_REV="ee1d15159cda6f3f97bfab9e487da146a8254970"
FFA_REQUIREMENT="ffa-fa3 @ git+https://github.com/demonatic/flash-attention.git@${FFA_REV}#subdirectory=hopper"
SUPPORTED_HEAD_DIMS=(64 96 128 192 256)
HEAD_DIMS_CSV="128"
NFUNC_CSV="1,3,5"
DTYPES_CSV="bf16,fp16"
DISABLE_FP16="FALSE"
MAX_JOBS_VALUE="2"
NVCC_THREADS_VALUE="4"
SPLIT_COMPILE_VALUE="32"
PRINT_CONFIG="FALSE"

usage() {
  cat <<'EOF'
Usage: scripts/kernel/install_magi_sm90.sh [options]

Install the verified MagiAttention SM90 CUTLASS overlay. The default matrix is
BF16+FP16, hdim128, nfunc 1/3/5, MAX_JOBS=2, NVCC_THREADS=4, and
--split-compile=32.

Options:
  --dim LIST            Compile comma-separated head-dimension buckets.
                        Supported values: 64,96,128,192,256. Default: 128.
  --nfunc LIST          Compile comma-separated odd nfunc values in [1, 33].
                        Default: 1,3,5.
  --dtype LIST          Enable bf16 or bf16,fp16 inputs. Default: bf16,fp16.
  --max-jobs N          Set parallel Ninja jobs. Default: 2.
  --nvcc-threads N      Set NVCC worker threads per job. Default: 4.
  --split-compile N     Set NVCC --split-compile concurrency. Default: 32.
  --print-config        Print the resolved matrix without checking CUDA or building.
  -h, --help            Show this help.

Non-default matrices are forwarded to the pinned upstream build and may fail
to compile. In particular, hdim64 and hdim256 arbitrary kernels exceed CUDA 13
register-allocation limits with the current upstream generator.
EOF
}

require_value() {
  if (( $# < 2 )); then
    echo "Option $1 requires a value." >&2
    usage >&2
    exit 2
  fi
}

validate_positive_integer() {
  local option="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${option} requires a positive integer, got '${value}'." >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --dim|--head-dim)
      require_value "$@"
      HEAD_DIMS_CSV="$2"
      shift 2
      ;;
    --nfunc)
      require_value "$@"
      NFUNC_CSV="$2"
      shift 2
      ;;
    --dtype)
      require_value "$@"
      DTYPES_CSV="$2"
      shift 2
      ;;
    --max-jobs)
      require_value "$@"
      MAX_JOBS_VALUE="$2"
      shift 2
      ;;
    --nvcc-threads)
      require_value "$@"
      NVCC_THREADS_VALUE="$2"
      shift 2
      ;;
    --split-compile)
      require_value "$@"
      SPLIT_COMPILE_VALUE="$2"
      shift 2
      ;;
    --print-config)
      PRINT_CONFIG="TRUE"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

validate_positive_integer "--max-jobs" "${MAX_JOBS_VALUE}"
validate_positive_integer "--nvcc-threads" "${NVCC_THREADS_VALUE}"
validate_positive_integer "--split-compile" "${SPLIT_COMPILE_VALUE}"

case "${DTYPES_CSV}" in
  bf16)
    DISABLE_FP16="TRUE"
    ;;
  bf16,fp16|fp16,bf16)
    DTYPES_CSV="bf16,fp16"
    DISABLE_FP16="FALSE"
    ;;
  fp16)
    echo "--dtype fp16 is unsupported because the pinned upstream build cannot disable BF16." >&2
    exit 2
    ;;
  *)
    echo "--dtype supports 'bf16' or 'bf16,fp16', got '${DTYPES_CSV}'." >&2
    exit 2
    ;;
esac

declare -A SELECTED_HEAD_DIMS=()
IFS=',' read -r -a REQUESTED_HEAD_DIMS <<< "${HEAD_DIMS_CSV}"
for dim in "${REQUESTED_HEAD_DIMS[@]}"; do
  case "${dim}" in
    64|96|128|192|256)
      SELECTED_HEAD_DIMS["${dim}"]=1
      ;;
    *)
      echo "--dim supports only 64,96,128,192,256, got '${dim}'." >&2
      exit 2
      ;;
  esac
done

HEAD_DIMS=()
for dim in "${SUPPORTED_HEAD_DIMS[@]}"; do
  if [[ -n "${SELECTED_HEAD_DIMS[${dim}]:-}" ]]; then
    HEAD_DIMS+=("${dim}")
  fi
done
if (( ${#HEAD_DIMS[@]} == 0 )); then
  echo "--dim must select at least one head-dimension bucket." >&2
  exit 2
fi
HEAD_DIMS_CSV="$(IFS=,; printf '%s' "${HEAD_DIMS[*]}")"
DIM_TAG="$(IFS=x; printf '%s' "${HEAD_DIMS[*]}")"

declare -A SELECTED_NFUNC=()
IFS=',' read -r -a REQUESTED_NFUNC <<< "${NFUNC_CSV}"
for nfunc in "${REQUESTED_NFUNC[@]}"; do
  if [[ ! "${nfunc}" =~ ^([1-9]|[1-2][0-9]|3[0-3])$ ]] || (( 10#${nfunc} % 2 == 0 )); then
    echo "--nfunc requires odd integers in [1, 33], got '${nfunc}'." >&2
    exit 2
  fi
  SELECTED_NFUNC["${nfunc}"]=1
done

NFUNC_VALUES=()
for (( nfunc = 1; nfunc <= 33; nfunc += 2 )); do
  if [[ -n "${SELECTED_NFUNC[${nfunc}]:-}" ]]; then
    NFUNC_VALUES+=("${nfunc}")
  fi
done
if (( ${#NFUNC_VALUES[@]} == 0 )); then
  echo "--nfunc must select at least one value." >&2
  exit 2
fi
NFUNC_CSV="$(IFS=,; printf '%s' "${NFUNC_VALUES[*]}")"
NFUNC_TAG="$(IFS=x; printf '%s' "${NFUNC_VALUES[*]}")"

DISABLE_HDIM64="TRUE"
DISABLE_HDIM96="TRUE"
DISABLE_HDIM128="TRUE"
DISABLE_HDIM192="TRUE"
DISABLE_HDIM256="TRUE"
for dim in "${HEAD_DIMS[@]}"; do
  case "${dim}" in
    64) DISABLE_HDIM64="FALSE" ;;
    96) DISABLE_HDIM96="FALSE" ;;
    128) DISABLE_HDIM128="FALSE" ;;
    192) DISABLE_HDIM192="FALSE" ;;
    256) DISABLE_HDIM256="FALSE" ;;
  esac
done

if [[ "${DISABLE_FP16}" == "TRUE" ]]; then
  DTYPE_TAG="bf16"
  DTYPE_LABEL="BF16"
else
  DTYPE_TAG="fp16bf16"
  DTYPE_LABEL="BF16+FP16"
fi
LOCAL_VERSION="sm90.${DTYPE_TAG}.hdim${DIM_TAG}.nfunc${NFUNC_TAG}.split${SPLIT_COMPILE_VALUE}"

print_config() {
  cat <<EOF
dtypes=${DTYPE_LABEL}
head_dim_buckets=${HEAD_DIMS_CSV}
nfunc=${NFUNC_CSV}
max_jobs=${MAX_JOBS_VALUE}
nvcc_threads=${NVCC_THREADS_VALUE}
split_compile=${SPLIT_COMPILE_VALUE}
local_version=${LOCAL_VERSION}
EOF
}

if [[ "${PRINT_CONFIG}" == "TRUE" ]]; then
  print_config
  exit 0
fi

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
  FFA_REV="${FFA_REV}" \
    LOCAL_VERSION="${LOCAL_VERSION}" \
    DISABLE_FP16="${DISABLE_FP16}" \
    HEAD_DIMS_CSV="${HEAD_DIMS_CSV}" \
    NFUNC_CSV="${NFUNC_CSV}" \
    "${PYTHON}" - <<'PY'
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

compiled_head_dims = {int(value) for value in os.environ["HEAD_DIMS_CSV"].split(",")}
expected_flags = {
    "FLASHATTENTION_DISABLE_BACKWARD": False,
    "FLASHATTENTION_DISABLE_SPLIT": True,
    "FLASHATTENTION_DISABLE_PAGEDKV": True,
    "FLASHATTENTION_DISABLE_APPENDKV": True,
    "FLASHATTENTION_DISABLE_LOCAL": True,
    "FLASHATTENTION_DISABLE_SOFTCAP": True,
    "FLASHATTENTION_DISABLE_PACKGQA": True,
    "FLASHATTENTION_DISABLE_FP16": os.environ["DISABLE_FP16"] == "TRUE",
    "FLASHATTENTION_DISABLE_FP8": True,
    "FLASHATTENTION_DISABLE_VARLEN": True,
    "FLASHATTENTION_DISABLE_CLUSTER": True,
    "FLASHATTENTION_DISABLE_HDIM64": 64 not in compiled_head_dims,
    "FLASHATTENTION_DISABLE_HDIM96": 96 not in compiled_head_dims,
    "FLASHATTENTION_DISABLE_HDIM128": 128 not in compiled_head_dims,
    "FLASHATTENTION_DISABLE_HDIM192": 192 not in compiled_head_dims,
    "FLASHATTENTION_DISABLE_HDIM256": 256 not in compiled_head_dims,
    "FLASHATTENTION_DISABLE_SM8x": True,
    "FLASHATTENTION_DISABLE_SM90": False,
    "FLASHATTENTION_ENABLE_VCOLMAJOR": False,
    "FLASH_ATTENTION_DISABLE_HDIMDIFF64": True,
    "FLASH_ATTENTION_DISABLE_HDIMDIFF192": True,
    "FLASHATTENTION_DISABLE_ARBITRARY": False,
    "FLASHATTENTION_NUM_FUNC": [int(value) for value in os.environ["NFUNC_CSV"].split(",")],
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
# upstream nfunc generator still instantiates some disabled feature
# combinations inside its nfunc translation units.
export FLASH_ATTENTION_DISABLE_APPENDKV="TRUE"
export FLASH_ATTENTION_DISABLE_ARBITRARY="FALSE"
export FLASH_ATTENTION_DISABLE_BACKWARD="FALSE"
export FLASH_ATTENTION_DISABLE_CLUSTER="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM64="${DISABLE_HDIM64}"
export FLASH_ATTENTION_DISABLE_HDIM96="${DISABLE_HDIM96}"
export FLASH_ATTENTION_DISABLE_HDIM128="${DISABLE_HDIM128}"
export FLASH_ATTENTION_DISABLE_HDIM192="${DISABLE_HDIM192}"
export FLASH_ATTENTION_DISABLE_HDIM256="${DISABLE_HDIM256}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIMDIFF192="TRUE"
export FLASH_ATTENTION_DISABLE_LOCAL="TRUE"
export FLASH_ATTENTION_DISABLE_PACKGQA="TRUE"
export FLASH_ATTENTION_DISABLE_PAGEDKV="TRUE"
export FLASH_ATTENTION_DISABLE_FP16="${DISABLE_FP16}"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
export FLASH_ATTENTION_DISABLE_SM80="TRUE"
export FLASH_ATTENTION_DISABLE_SM90="FALSE"
export FLASH_ATTENTION_ENABLE_VCOLMAJOR="FALSE"
export FLASH_ATTENTION_DISABLE_SOFTCAP="TRUE"
export FLASH_ATTENTION_DISABLE_SPLIT="TRUE"
export FLASH_ATTENTION_DISABLE_VARLEN="TRUE"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
# MagiAttention imports the pinned overlay's unstable API symbols directly.
export FLASH_ATTENTION_FORCE_UNSTABLE_API="TRUE"
export FLASH_ATTENTION_NUM_FUNC="${NFUNC_CSV}"
export FLASH_ATTN_LOCAL_VERSION="${LOCAL_VERSION}"
export MAX_JOBS="${MAX_JOBS_VALUE}"
export NVCC_THREADS="${NVCC_THREADS_VALUE}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+${NVCC_APPEND_FLAGS} }--split-compile=${SPLIT_COMPILE_VALUE}"

echo "Installing MagiAttention CUTLASS FFA for ${VISIBLE_ARCHS} with CUDA ${NVCC_VERSION}."
echo "Requested runtime matrix: ${DTYPE_LABEL}, hdim=${HEAD_DIMS_CSV}, nfunc=${NFUNC_CSV}."
echo "Compiler parallelism: MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS}, split-compile=${SPLIT_COMPILE_VALUE}."
if [[ -n "${SELECTED_HEAD_DIMS[64]:-}" || -n "${SELECTED_HEAD_DIMS[256]:-}" ]]; then
  echo "Warning: hdim64/256 arbitrary kernels may exceed CUDA 13 register-allocation limits." >&2
fi
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
