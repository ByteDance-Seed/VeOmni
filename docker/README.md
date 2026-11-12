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

## CANN 9.2.1 A2/A3 images

These standalone Dockerfiles use Python 3.12, torch 2.10.0,
torch-npu 2.10.0.post7 and Transformers 5.16.1. Select the Dockerfile for
both the NPU generation and the host architecture:

| Target | Platform | Dockerfile | Triton | Triton-Ascend |
| --- | --- | --- | --- | --- |
| A2 ARM | `linux/arm64` | [910b.arm](ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.arm) | 3.6.0 | 3.6.0 |
| A2 x86 | `linux/amd64` | [910b.x86](ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.x86) | 3.2.0 | 3.2.2 |
| A3 ARM | `linux/arm64` | [a3](ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a3) | 3.5.0 | 3.2.2 |

A2 uses the `quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12` base;
A3 uses `quay.io/ascend/cann:9.2.1-a3-ubuntu22.04-py3.12`.
The images install into system Python. ARM variants select `npu_aarch64`
and build TorchCodec from source; x86 selects `npu`.
FLA source is pinned to `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`, with
the package and SoC selected for A2 or A3.

Build from the repository root on a matching host architecture. For example,
to build A2 x86:

```bash
docker build --platform linux/amd64 \
  -f docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_910b.x86 \
  --build-arg FLA_BUILD_JOBS=4 \
  -t veomni:cann921-a2-x86 .
```

For ARM, use `--platform linux/arm64` and the corresponding Dockerfile above.
Triton is installed before Triton-Ascend because the Ascend package overlays
shared Triton files. The Dockerfiles retain wheel files under `/app` and
record wheel checksums and installed package versions for each build.

### Candidate builders

The candidate workflows build and publish images to GHCR:

| Workflow | Platform | Image tag under `ghcr.io/<owner>/veomni` |
| --- | --- | --- |
| [A2 builder](../.github/workflows/build-cann921-a2.yml) | `linux/amd64` | `cann921-a2-x86-<commit>` |
| [A3 builder](../.github/workflows/build-cann921-a3.yml) | `linux/arm64` | `cann921-a3-arm64-<commit>-<run-id>-<attempt>` |

Both declare `workflow_dispatch` and path-filtered pushes on
`zpl1n/ascend-cann921-a2-a3`. A2 watches its workflow and `docker/ascend/**`;
A3 watches its workflow and its A3 Dockerfile. There is no A2 ARM candidate
workflow in this addition. These builders are separate from the existing
Quay publishing workflows.

### Validation scope

The candidate builders perform image construction and publication, including
the Dockerfiles' package-version and import checks. They do not run NPU
numerical tests. The existing NPU unit/e2e workflows retain their upstream
image and dependency setup; they do not validate these candidate images.

Device validation remains a local procedure. Keep its scripts, logs and
environment-specific patches outside the committed source tree. Record the
exact image, architecture and package versions when interpreting results;
results from a modified local image do not validate a different Dockerfile.

An image's preinstalled system packages are separate from the project
environment created by `uv sync`. Local validation must explicitly select
which environment it tests. Any wheel overlays applied after synchronization
are additional environment changes, not part of the lockfile alone.
