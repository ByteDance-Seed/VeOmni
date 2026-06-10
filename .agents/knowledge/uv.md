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
│   │                  + FA2 / FA3 / FA4 / FlashQLA (all source-built)
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
│   ├── no-build-isolation-package  flash-attn, flash-attn-3
│   └── sources              Custom indexes, direct wheel URLs (av, torch),
│                            git sources (flash-attn-3 / flash-attn-4 / flash-qla)
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

A fresh `--extra gpu` source-builds FA2/FA3/FA4/FlashQLA (~60–90 min total).
uv caches the wheels under `~/.cache/uv` for subsequent syncs.

## Transformers Version

`transformers==5.9.0` is pinned by the `transformers-stable` group (in
`default-groups`). Kept out of `[project.dependencies]` so pip users are not
forced into a specific 5.x patch.

## torch Source Pinning

- **GPU**: direct wheel URL (not the pytorch index) — avoids uv resolving
  cu128_full wheels that drop nvidia-* deps.
- **NPU**: pytorch index (`https://download.pytorch.org/whl/`).

## Source-Built Attention Kernels

| Package | Source | `no-build-isolation-package` |
|---|---|---|
| `flash-attn` (FA2) | PyPI sdist | yes |
| `flash-attn-3` (Hopper) | git: Dao-AILab/flash-attention/hopper | yes |
| `flash-attn-4` (cute) | git: Dao-AILab/flash-attention/flash_attn/cute | no |
| `flash-qla` | git: QwenLM/FlashQLA | no |

Three pyproject knobs make a fresh `uv sync --extra gpu` succeed:

1. **`[[tool.uv.dependency-metadata]]`** with `version` for `flash-attn-3`
   and `flash-qla`. They have no `pyproject.toml`; without static metadata
   uv runs their setup.py on a fresh venv and crashes with
   `ModuleNotFoundError: No module named 'setuptools'`.

2. **`[tool.uv.extra-build-dependencies]`** seeds `setuptools / wheel /
   packaging / ninja` (+ `torch` where needed) — uv venvs are not seeded.

3. **`FLASH_ATTENTION_FORCE_BUILD=TRUE`** (env var) forces FA2/FA3 setup.py
   to source-build instead of guessing a github prebuilt wheel URL (no cu13
   variants exist). Set in the cuda Dockerfile; export locally.

## Common Commands

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE uv sync --extra gpu --dev   # local dev
uv lock                                                       # after pyproject edits
uv sync --locked --all-packages --extra gpu --dev            # docker / CI
```

## Key Rules

1. **Always commit `uv.lock` with `pyproject.toml`** — Docker uses `--locked`.
2. **torch bumps touch 4+ places** (extras, overrides, sources wheel URL).
3. **All four flash-attn/qla kernels compile from source every fresh sync.**
   `FLASH_ATTENTION_FORCE_BUILD=TRUE` is required for FA2/FA3.
4. **uv bumps require Docker rebuilds**; concrete pins must stay in range.
5. **`override-dependencies` `extra == '...'` markers are load-bearing.**
6. **`transformers==5.9.0` is the only supported version.** New code targets
   v5 + FSDP2 + patchgen-generated modeling.
