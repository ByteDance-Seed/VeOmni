# uv Dependency Management

VeOmni uses [uv](https://docs.astral.sh/uv/) for dependency management. This document describes the architecture.

## uv Version

Pinned to a specific version for reproducibility. **Three locations must stay in sync:**

| Location | Format |
|----------|--------|
| `pyproject.toml` -> `[tool.uv]` -> `required-version` | `"==X.Y.Z"` |
| `docker/cuda/Dockerfile.cu129` | `COPY --from=ghcr.io/astral-sh/uv:X.Y.Z` |
| `docker/ascend/Dockerfile.*` | same pattern |

## Dependency Layout

```
pyproject.toml
├── [project.dependencies]              Core deps (always installed, transformers NOT included here)
├── [project.optional-dependencies]     Hardware & feature extras
│   ├── gpu          torch 2.9.1+cu129, flash-attn, liger-kernel, etc.
│   ├── npu          torch 2.7.1+cpu, torch-npu
│   ├── npu_aarch64  torch 2.7.1 (native)
│   ├── audio/video  torchcodec, av, librosa, soundfile
│   ├── dit          diffusers, av, peft
│   ├── megatron     megatron-energon
│   ├── trl          trl
│   ├── fa4          flash-attn-4, nvidia-cutlass-dsl
│   ├── transformers5-exp   transformers==5.2.0 (opt-in v5)
│   └── dev          pre-commit, ruff, pytest (legacy pip compat)
├── [dependency-groups]                 Dev-only (uv-native)
│   ├── dev                  includes lint + test
│   ├── lint                 pre-commit, ruff
│   ├── test                 pytest, expecttest, rich
│   └── transformers-stable  transformers==4.57.3 (default, in default-groups)
├── [tool.uv]
│   ├── required-version     Pinned uv version
│   ├── override-dependencies  Per-extra torch/CUDA pins + cudnn override
│   ├── conflicts            gpu/npu mutual exclusion + transformers-stable/transformers5-exp
│   ├── no-build-isolation-package  flash-attn, flash-attn-3
│   └── sources              Custom indexes and direct wheel URLs
└── uv.lock                  Lockfile (committed, used by Docker --locked)
```

## Hardware Extras (Mutually Exclusive)

`gpu`, `npu`, and `npu_aarch64` are declared as conflicts — only one can be installed at a time.

```bash
uv sync --extra gpu --dev           # NVIDIA GPU
uv sync --extra npu --dev           # Ascend NPU (x86)
uv sync --extra npu_aarch64 --dev   # Ascend NPU (ARM)
```

## Transformers Version (Dual-Track)

VeOmni supports two mutually exclusive transformers versions via uv conflicts:

| Track | Mechanism | Version | How to activate |
|-------|-----------|---------|-----------------|
| **Default (stable)** | Dependency group `transformers-stable` (in `default-groups`) | `4.57.3` | `uv sync --extra gpu --dev` (automatic) |
| **Experimental (v5)** | Optional extra `transformers5-exp` | `5.2.0` | `uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --dev` |

`transformers-stable` (group) and `transformers5-exp` (extra) are declared as conflicts in `[tool.uv.conflicts]`. **All new development should target v5.** The v4 stable track will be removed once migration is complete.

## torch Source Pinning

torch, torchvision, torchaudio use custom sources:

- **GPU**: torch uses a direct wheel URL (not the pytorch index) to avoid uv resolving to incompatible cu128_full wheels. The URL must be updated manually when bumping torch.
- **NPU**: uses the `pytorch` index (`https://download.pytorch.org/whl/`)
- **flash-attn / flash-attn-3**: direct wheel URLs tied to specific torch+CUDA combinations. Listed under `no-build-isolation-package`.

## Common Commands

```bash
# Initial setup (default transformers 4.57.3)
uv sync --extra gpu --extra audio --dev

# Switch to transformers 5.2.0 (for new development)
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --dev

# Regenerate lockfile after pyproject.toml changes
uv lock

# Sync after lockfile update
uv sync --extra gpu --dev

# Docker builds (CI)
uv sync --locked --all-packages --extra gpu --dev
```

## Key Rules

1. **Always commit `uv.lock` with `pyproject.toml`** — Docker builds use `--locked`.
2. **torch version changes touch 4+ places** in pyproject.toml (extras, overrides, sources, wheel URL).
3. **flash-attn wheels are torch-version-specific** — bumping torch requires new wheels.
4. **uv version changes require Docker rebuilds** — update Dockerfiles and release new images.
5. **`override-dependencies` markers are load-bearing** — the `extra == 'gpu'` guards prevent uv from downloading wrong torch variants.
6. **`transformers-stable` and `transformers5-exp` are mutually exclusive** — uv conflicts enforce this. Never install both. All new code must work with v5; use `is_transformers_version_greater_or_equal_to()` for v4 compat guards.
