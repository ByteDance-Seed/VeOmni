# Docker

This directory ships the Dockerfiles used to build VeOmni runtime images. Every
Dockerfile is standalone and hand-maintained — pick the one matching your
accelerator and edit it directly.

```
docker/
├── cuda/                  NVIDIA images (NGC PyTorch base)
│   └── Dockerfile.cu130
├── ascend/                Ascend NPU images (CANN base), per CANN version / SoC / arch
│   ├── Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.x86
│   ├── Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.arm
│   ├── Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3
│   └── ...
└── rocm/                  AMD images (ROCm Primus base)
    └── Dockerfile.ROCm7.14
```

Ascend file names encode the variant: CANN version, torch-npu version where it
is pinned, then the SoC (`910b` / A2 vs `a3`) and the host arch (`.x86` /
`.arm`; the `a3` files are arm-only).

## Building

The build context is the repo root — the images `COPY` the checkout in and
install from `pyproject.toml` / `uv.lock`:

```bash
docker build -f docker/cuda/Dockerfile.cu130 -t veomni:cu130 .
```

Most images expose `APT_SOURCE` and `PIP_INDEX` build args so you can swap in
closer mirrors without editing the Dockerfile:

```bash
docker build -f docker/cuda/Dockerfile.cu130 \
    --build-arg APT_SOURCE=https://mirrors.example.com/ubuntu/ \
    --build-arg PIP_INDEX=https://mirrors.example.com/pypi/simple \
    -t veomni:cu130 .
```

## Adding or updating an image

- Name the file `Dockerfile.<variant>` under the vendor directory, and keep the
  provenance comments above `FROM` (base image page, release notes, upstream
  torch-npu repo) so the pinned base can be traced.
- The `gpu`, `npu` and `npu_aarch64` extras are mutually exclusive (see
  `[tool.uv].conflicts` in `pyproject.toml`), so each image selects exactly one
  instead of using `--all-extras`.
- uv-based images pin uv through `COPY --from=ghcr.io/astral-sh/uv:<version>`;
  keep that version inside the `[tool.uv].required-version` range declared in
  `pyproject.toml`.
- Some CANN bases cannot validate the github.com / pythonhosted.org certificates
  that `uv sync --locked` revalidates; those images pass
  `--allow-insecure-host github.com --allow-insecure-host pythonhosted.org`
  (see `Dockerfile.ascend_8.3.rc2_a2.x86` and `Dockerfile.ascend_9.0.0_a2.x86`).
- [`docker-build-ascend-a2.yml`](../.github/workflows/docker-build-ascend-a2.yml)
  and [`docker-build-ascend-a3.yml`](../.github/workflows/docker-build-ascend-a3.yml)
  build and push a fixed set of `ascend/` Dockerfiles on merge to `main`; check
  those workflows before renaming or removing a file under `ascend/`.

## CANN 9.2.1 A2/A3 candidates

These standalone candidates use torch 2.10.0, torch-npu 2.10.0.post7,
Triton/Triton-Ascend 3.6.0, Transformers 5.16.1 and Python 3.12.
The A3 Dockerfile is copied from `xssty123/VeOmni` commit
`298394aee6806472065142d86539308ed36b1101` (`codex/ascend-cann921`).
A2 uses the same packaging procedure with `FLA_NPU_SOC=ascend910b` and
`flash-linear-attention-npu-a2`; A3 uses `ascend910_93` and the `-a3` wheel.
Both pin FLA source to `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`.

| Target | Platform | Dockerfile suffix | Base image tag |
| --- | --- | --- | --- |
| A2 ARM | linux/arm64 | 910b.arm | 9.2.1-910b-ubuntu22.04-py3.12 |
| A2 x86 | linux/amd64 | 910b.x86 | 9.2.1-910b-ubuntu22.04-py3.12 |
| A3 | linux/arm64 | a3 | 9.2.1-a3-ubuntu22.04-py3.12 |

All bases are under `quay.io/ascend/cann`. ARM images install the
`npu_aarch64` extra and build TorchCodec v0.10.0 from source. A2 x86 installs
`npu` with the PyTorch CPU wheel index and uses TorchCodec 0.10.0's binary wheel.
These candidates install into system Python as root. A2 x86 therefore differs
from the older 9.1.0 image's tiger user and `/app/.venv` layout.

From a clean committed checkout, build on the matching host architecture:

```bash
# Choose a2 / 910b.arm / linux/arm64, a2 / 910b.x86 / linux/amd64,
# or a3 / a3 / linux/arm64.
target=a2
variant=910b.arm
platform=linux/arm64
revision=$(git rev-parse HEAD)
image=veomni:cann921-${variant}
docker build --platform "$platform" \
  -f "docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_${variant}" \
  --label "org.opencontainers.image.revision=$revision" \
  --label "ai.veomni.docker-revision=$revision" \
  --label ai.veomni.fla-revision=a7c608f2e3f828779ceeec1fcf155a6e37ede83c \
  --label "ai.veomni.target=$target" \
  --build-arg FLA_BUILD_JOBS=4 -t "$image" .

mkdir -p reports
docker image inspect "$image" > reports/image-inspect.json
docker run --rm --entrypoint /bin/bash \
  -v "$PWD/reports:/reports" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e TARGET="$target" -e REVISION="$revision" "$image" -c '
    set -eo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python /app/docker/ascend/tests/ascend_image_smoke.py \
      --image-inspect /reports/image-inspect.json \
      --expected-docker-commit "$REVISION" --target "$TARGET" \
      --output /reports/smoke.json
  '
```

The smoke validates the image architecture, Python 3.12, CANN, package versions,
retained wheel checksums, FLA Stable ABI, and library imports without NPU execution.
It accepts Python 3.12 patch updates, unlike the source branch's exact 3.12.16
check. `pip check` results are recorded separately: a passed packaging smoke
can still have `dependency_consistent=false` (including the project's original
torch-npu pin versus the image's post7 override). Inspect both fields before use.

The retained `/app/<target>-ci-wheels.sha256` and wheels can be used to restore
these overrides after creating a separate CI environment. A new `uv sync` does
not validate the preinstalled image and can reset the selected package versions.
Base tags and unpinned build dependencies can change between builds; freeze files
and checksums record a particular build, not a fully locked dependency solution.

These additions do not change the existing automatic Quay publishing workflows.
The source fork's private GHCR candidate workflows and A5 diagnostics are not
included. Full image builds, packaging smoke and target-device numerical tests
must pass before these candidates are treated as validated runtime images.
