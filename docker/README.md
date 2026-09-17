# Docker

This directory ships the Dockerfiles used to build VeOmni runtime images. Every
Dockerfile is standalone and hand-maintained — pick the one matching your
accelerator and edit it directly.

```
docker/
├── cuda/                  NVIDIA images (NGC PyTorch base)
│   └── Dockerfile.cu130       multi-arch: amd64 + arm64
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

`cuda/Dockerfile.cu130` carries no arch suffix because it builds both.

## Grace (GB200 / GH200)

Those queues require an ARM image and reject an x86 one before it starts. The
same `cuda/Dockerfile.cu130` covers them — add `linux/arm64` to the build's
platform list, no separate file:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
    -f docker/cuda/Dockerfile.cu130 -t veomni:cu130 .
```

Two things make that work, and both are easy to undo by accident:

- **Nothing pins a platform.** `FROM` takes no `--platform`, so buildx resolves
  the multi-arch NGC manifest per target. Adding `--platform=linux/arm64` there
  would pin *every* arch of the build to arm64.
- **`APT_SOURCE` defaults to empty**, leaving whichever sources.list the base
  image ships (amd64 `archive.ubuntu.com`, arm64 `ports.ubuntu.com`). A mirror
  passed through that arg must serve the target architecture.

One runtime difference is load-bearing: **FA2 has no aarch64 wheel**, and
`pyproject.toml` scopes `flash-attn` to `x86_64`. A Grace node therefore has
CUDA but no `flash_attn`, so model code must branch on
`is_flash_attn_2_available()` rather than on `IS_CUDA_AVAILABLE` — the latter
turns a missing wheel into an import error that takes the whole package down.

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
