# uv Dependency Management

VeOmni uses [uv](https://docs.astral.sh/uv/) for dependency management. This document describes the architecture.

## uv Version

`pyproject.toml` declares a **range** (`>=0.9.8,<0.12`); the Dockerfiles and
CI install a concrete pin and use `--locked` / `--frozen` for reproducibility.
**Every concrete uv pin must stay inside the pyproject range.**

| Location | Format |
|----------|--------|
| `pyproject.toml` -> `[tool.uv]` -> `required-version` | range |
| `docker/cuda/Dockerfile.cu130`, `docker/ascend/Dockerfile.*` | `COPY --from=ghcr.io/astral-sh/uv:X.Y.Z` |
| `.github/workflows/check_patchgen.yml` | `setup-uv` `version: "X.Y.Z"` |

## Dependency Layout

```
pyproject.toml
├── [project.dependencies]              Core deps (always installed, transformers NOT included here)
├── [project.optional-dependencies]     Hardware-shaped extras (deliberately just three + legacy `dev`)
│   ├── gpu          NVIDIA x86_64 — full superset:
│   │                  torch 2.11.0+cu130 + cu130 nvidia stack + cuda-python
│   │                  + FA2 (cp311/cp312 wheels) / FA3 (sm90 abi3 wheel)
│   │                  + FA4 / FlashQLA (source-built from git)
│   │                  + liger-kernel + FLA + quack + DLPack ext
│   │                  + diffusers / av / librosa / soundfile / ftfy / peft
│   │                  + megatron-energon (optional dataset format)
│   ├── npu          Ascend NPU x86_64 — full superset, minus CUDA-only kernels:
│   │                  torch 2.7.1+cpu + torch-npu + diffusers / av / peft / megatron-energon
│   ├── npu_aarch64  Ascend NPU aarch64 — minimal (torch + torch-npu only;
│   │                  av/torchcodec lack pinned aarch64 wheels)
│   └── dev          pre-commit, ruff, pytest (legacy pip-style; modern uv path is the dev group)
├── [dependency-groups]                 Dev-only (uv-native)
│   ├── dev                  includes lint + test + patchgen
│   ├── lint                 pre-commit, ruff
│   ├── test                 pytest, expecttest, rich
│   ├── patchgen             patchgen (path source under patchgen-pkg/)
│   └── transformers-stable  transformers==5.9.0 (default, in default-groups)
├── [tool.uv]
│   ├── required-version     Pinned uv version
│   ├── override-dependencies  Per-extra torch/CUDA pins (markers scoped to gpu/npu/npu_aarch64)
│   ├── conflicts            gpu/npu/npu_aarch64 mutual exclusion
│   └── sources              Custom indexes, direct wheel URLs (av, torch,
│                            FA2 cp311/cp312, FA3 sm90 abi3); git sources
│                            (flash-attn-4 / flash-qla)
└── uv.lock                  Lockfile (committed, used by Docker --locked)
```

## Hardware Extras (Mutually Exclusive)

`gpu` / `npu` / `npu_aarch64` are declared as conflicts. trl is not included
— VeOmni's DPO trainer is from-scratch.

```bash
uv sync --extra gpu --dev           # NVIDIA GPU
uv sync --extra npu --dev           # Ascend NPU x86
uv sync --extra npu_aarch64 --dev   # Ascend NPU ARM (minimal)
```

A fresh `--extra gpu` installs FA2/FA3 from prebuilt wheels (cp311 + cp312)
and source-builds only FA4 + FlashQLA (~10–20 min total). uv caches built
wheels under `~/.cache/uv`.

## Transformers Version

`transformers==5.9.0` is pinned by the `transformers-stable` group (in
`default-groups`). Kept out of `[project.dependencies]` so pip users are not
forced into a specific 5.x patch.

## torch Source Pinning

- **GPU**: direct wheel URL (not the pytorch index) — avoids uv resolving
  cu128_full wheels that drop nvidia-* deps.
- **NPU**: pytorch index (`https://download.pytorch.org/whl/`).

## Attention Kernels

| Package | Source | Notes |
|---|---|---|
| `flash-attn` (FA2) | cp311 wheel (v0.0.3) + cp312 wheel (v0.0.5), Luosuu cu130/torch2.11/sm80-100 | one wheel per Python minor |
| `flash-attn-3` (Hopper) | cp310-abi3 wheel (v0.0.5), Luosuu cu130/torch2.11/sm90 | abi3 covers cp310+ |
| `flash-attn-4` (cute) | git: Dao-AILab/flash-attention/flash_attn/cute | source-built |
| `flash-qla` | git: QwenLM/FlashQLA | source-built |

Two pyproject knobs make source builds (FA4, FlashQLA) succeed:

1. **`[[tool.uv.dependency-metadata]]`** with `version` for `flash-qla`.
   It has no `pyproject.toml`; without static metadata uv runs its setup.py
   on a fresh venv and crashes with `ModuleNotFoundError: No module named
   'setuptools'`. The static `requires-dist` mirrors flash-qla's own
   install_requires (`torch`, `tilelang==0.1.8`, `apache-tvm-ffi==0.1.9`)
   — `flash_qla/__init__.py` top-level imports `tilelang`, so they have to
   ship alongside, even though the `flash_qla` kernel itself only binds on
   sm90 (gated by `KernelSpec(min_compute_capability=90)`).

2. **`[tool.uv.extra-build-dependencies]`** seeds `setuptools / wheel /
   packaging / ninja` (+ `torch` where needed) — uv venvs are not seeded.

`FLASH_ATTENTION_FORCE_BUILD=TRUE` and `[tool.uv.no-build-isolation-package]`
are gone — no FA setup.py runs anywhere now (FA2/3 wheel, FA4 cute is a
DSL package, flash-qla uses dependency-metadata).

## Common Commands

```bash
uv sync --extra gpu --dev                          # local dev (cp311 or cp312)
uv lock                                             # after pyproject edits
uv sync --locked --all-packages --extra gpu --dev  # docker / CI
```

## Key Rules

1. **Always commit `uv.lock` with `pyproject.toml`** — Docker uses `--locked`.
2. **torch bumps touch 4+ places** (extras, overrides, sources wheel URL).
3. **FA2/FA3 prebuilt wheels are pinned to torch 2.11 cu130 cp311/cp312/abi3.**
   Bumping torch / Python / cuda requires a matching Luosuu release.
4. **uv bumps require Docker rebuilds**; concrete pins must stay in range.
5. **`override-dependencies` `extra == '...'` markers are load-bearing.**
6. **`transformers==5.9.0` is the only supported version.** New code targets
   v5 + FSDP2 + patchgen-generated modeling.
