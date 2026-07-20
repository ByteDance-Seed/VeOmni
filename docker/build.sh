#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
IMAGE_NAME=${VEOMNI_IMAGE:-veomni:a100-cu126-torch271-tf451}
FLASH_ATTN_WHEEL_NAME=flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
FLASH_ATTN_WHEEL_SHA256=2d1819070504b6f9d8f117c453815835c6730a90b01aa25dca13f6d2b53c3f1e
LOCAL_FLASH_ATTN_WHEEL=${VEOMNI_FLASH_ATTN_WHEEL:-/mgData4/datasets/cache/veomni-docker/wheels/${FLASH_ATTN_WHEEL_NAME}}
build_args=()
wheel_server_pid=

cleanup() {
    if [[ -n "${wheel_server_pid}" ]]; then
        kill "${wheel_server_pid}" >/dev/null 2>&1 || true
        wait "${wheel_server_pid}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

if [[ -n "${VEOMNI_FLASH_ATTN_WHEEL_URL:-}" ]]; then
    build_args+=(--build-arg "FLASH_ATTN_WHEEL_URL=${VEOMNI_FLASH_ATTN_WHEEL_URL}")
elif [[ -f "${LOCAL_FLASH_ATTN_WHEEL}" ]]; then
    actual_sha256=$(sha256sum "${LOCAL_FLASH_ATTN_WHEEL}" | awk '{print $1}')
    if [[ "${actual_sha256}" != "${FLASH_ATTN_WHEEL_SHA256}" ]]; then
        echo "FlashAttention wheel SHA-256 mismatch: ${LOCAL_FLASH_ATTN_WHEEL}" >&2
        exit 1
    fi

    wheel_server_port=${VEOMNI_WHEEL_SERVER_PORT:-18765}
    if [[ ! "${wheel_server_port}" =~ ^[0-9]+$ ]] || ((wheel_server_port < 1024 || wheel_server_port > 65535)); then
        echo "VEOMNI_WHEEL_SERVER_PORT must be an integer from 1024 through 65535." >&2
        exit 1
    fi
    wheel_server_dir=$(dirname -- "${LOCAL_FLASH_ATTN_WHEEL}")
    wheel_server_file=$(basename -- "${LOCAL_FLASH_ATTN_WHEEL}")
    wheel_server_file_url=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "${wheel_server_file}")
    wheel_server_url="http://127.0.0.1:${wheel_server_port}/${wheel_server_file_url}"
    python3 -m http.server "${wheel_server_port}" \
        --bind 127.0.0.1 \
        --directory "${wheel_server_dir}" \
        >/dev/null 2>&1 &
    wheel_server_pid=$!

    wheel_server_ready=false
    for _ in {1..50}; do
        if python3 -c 'import sys, urllib.request; urllib.request.urlopen(urllib.request.Request(sys.argv[1], method="HEAD"), timeout=1).close()' "${wheel_server_url}" >/dev/null 2>&1; then
            wheel_server_ready=true
            break
        fi
        if ! kill -0 "${wheel_server_pid}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
    if [[ "${wheel_server_ready}" != true ]]; then
        echo "Could not start the local FlashAttention wheel server." >&2
        exit 1
    fi

    echo "Using verified local FlashAttention wheel: ${LOCAL_FLASH_ATTN_WHEEL}"
    build_args+=(--build-arg "FLASH_ATTN_WHEEL_URL=${wheel_server_url}")
fi

if ! docker info >/dev/null 2>&1; then
    echo "Cannot access the Docker daemon. Re-login after joining the docker group, or run: newgrp docker" >&2
    exit 1
fi

docker build \
    --network host \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${IMAGE_NAME}" \
    "${build_args[@]}" \
    "${REPO_ROOT}"

echo "Built ${IMAGE_NAME}"
