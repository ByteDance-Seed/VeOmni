#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
IMAGE_NAME=${VEOMNI_IMAGE:-veomni:a100-cu126-torch271-tf451}
DATA_ROOT=${VEOMNI_DATA_ROOT:-/mgData4}
CACHE_ROOT=${VEOMNI_CACHE_ROOT:-${DATA_ROOT}/datasets/cache/veomni-docker}

if ! docker info >/dev/null 2>&1; then
    echo "Cannot access the Docker daemon. Re-login after joining the docker group, or run: newgrp docker" >&2
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Data root does not exist: ${DATA_ROOT}" >&2
    exit 1
fi

mkdir -p \
    "${CACHE_ROOT}/home" \
    "${CACHE_ROOT}/huggingface/datasets" \
    "${CACHE_ROOT}/cuda" \
    "${CACHE_ROOT}/torchinductor" \
    "${CACHE_ROOT}/torch-extensions" \
    "${CACHE_ROOT}/triton" \
    "${CACHE_ROOT}/xdg"

docker_args=(
    run
    --rm
    --gpus all
    --ipc host
    --network host
    --init
    --ulimit memlock=-1:-1
    --ulimit stack=67108864:67108864
    --user "$(id -u):$(id -g)"
    --env "HOME=${CACHE_ROOT}/home"
    --env "HF_HOME=${CACHE_ROOT}/huggingface"
    --env "HF_DATASETS_CACHE=${CACHE_ROOT}/huggingface/datasets"
    --env "CUDA_CACHE_PATH=${CACHE_ROOT}/cuda"
    --env "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor"
    --env "TORCH_EXTENSIONS_DIR=${CACHE_ROOT}/torch-extensions"
    --env "TRITON_CACHE_DIR=${CACHE_ROOT}/triton"
    --env "XDG_CACHE_HOME=${CACHE_ROOT}/xdg"
    --env "PYTHONPATH=/workspace/VeOmni"
    --env "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
    --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    --volume "${DATA_ROOT}:${DATA_ROOT}"
    --volume "${REPO_ROOT}:/workspace/VeOmni"
    --workdir /workspace/VeOmni
)

primary_gid=$(id -g)
for supplementary_gid in $(id -G); do
    if [[ "${supplementary_gid}" != "${primary_gid}" ]]; then
        docker_args+=(--group-add "${supplementary_gid}")
    fi
done

if [[ -t 0 && -t 1 ]]; then
    docker_args+=(--interactive --tty)
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker "${docker_args[@]}" "${IMAGE_NAME}" "$@"
