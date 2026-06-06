# uv Dependency Management

VeOmni uses [uv](https://docs.astral.sh/uv/) for dependency management. This document describes the architecture.

## uv Version

`pyproject.toml` constrains uv to a **range** (currently `>=0.9.8,<0.12`, i.e.
0.9.8 through 0.11.x) so local devs aren't forced onto one weekly uv build —
they're encouraged to stay reasonably current within it, and the window will be
tightened later. Reproducibility is preserved because every place that produces
or consumes the lockfile installs a **concrete**, in-range uv and never
re-resolves: the Dockerfiles `COPY` a fixed uv and `uv sync --locked`, the
container CI jobs `uv run --frozen`, and the `check_patchgen` CI job (which runs
on `ubuntu-latest`, not a prebuilt image) pins uv via `setup-uv`'s `version:`
input. **Every concrete uv pin must stay inside the pyproject range.**

| Location | Format |
|----------|--------|
| `pyproject.toml` -> `[tool.uv]` -> `required-version` | range, e.g. `">=0.9.8,<0.12"` |
| `docker/cuda/Dockerfile.cu130` | `COPY --from=ghcr.io/astral-sh/uv:X.Y.Z` (concrete, inside range) |
| `docker/ascend/Dockerfile.*` | same pattern |
| `.github/workflows/check_patchgen.yml` | `setup-uv` `version: "X.Y.Z"` (concrete, inside range) |

## Dependency Layout

```
pyproject.toml
├── [project.dependencies]              Core deps (always installed, transformers NOT included here)
├── [project.optional-dependencies]     Hardware-shaped extras (deliberately just three + legacy `dev`)
│   ├── gpu          NVIDIA x86_64 — full superset:
│   │                  torch 2.11.0+cu130 + cu130 nvidia stack + cuda-python
│   │                  + FA2 / FA3 / FA4 / FlashQLA (all source-built)
│   │                  + liger-kernel + FLA + quack + DLPack ext
│   │                  + diffusers / av / librosa / soundfile / ftfy / trl / peft
│   ├── npu          Ascend NPU x86_64 — full superset, minus CUDA-only kernels:
│   │                  torch 2.7.1+cpu + torch-npu + diffusers / av / trl / peft
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

History note: an earlier revision split the Python-level deps (`audio`, `video`,
`dit`, `trl`, `lora`, `fa3`, `fa4`, `flash-qla`, `megatron`) into nine separate
extras, which forced docker / CI / docs to chain seven-plus `--extra` flags.
Those have all been folded into `gpu` / `npu` so a typical install is a single
`--extra <gpu|npu|npu_aarch64>`.

## Hardware Extras (Mutually Exclusive)

`gpu`, `npu`, and `npu_aarch64` are declared as conflicts — only one can be
installed at a time. There are no other extras to chain alongside them; each
is a complete superset.

```bash
uv sync --extra gpu --dev           # NVIDIA GPU — torch+cu130, FA2/3/4/FlashQLA, diffusion, audio, trl, peft
uv sync --extra npu --dev           # Ascend NPU x86 — torch+cpu, torch-npu, diffusion, audio, trl, peft
uv sync --extra npu_aarch64 --dev   # Ascend NPU ARM — minimal (torch + torch-npu)
```

A fresh `uv sync --extra gpu` source-builds flash-attn, flash-attn-3,
flash-attn-4, and flash-qla (~60–90 min combined the first time). uv caches
the built wheels under `~/.cache/uv`, so subsequent syncs and CI runners with
a populated cache are fast.

## Transformers Version

`transformers==5.9.0` is pinned by the `transformers-stable` dependency
group, which is listed in `[tool.uv] default-groups`. `uv sync` (no extra
flags) installs it automatically. We keep the version out of
`[project.dependencies]` so pip users are not forced into a specific 5.x
patch release; pip users should `pip install transformers==5.9.0` manually.

## torch Source Pinning

torch, torchvision, torchaudio use custom sources:

- **GPU**: torch uses a direct wheel URL (not the pytorch index) to avoid uv resolving to incompatible cu128_full wheels. The URL must be updated manually when bumping torch.
- **NPU**: uses the `pytorch` index (`https://download.pytorch.org/whl/`)
- **flash-attn / flash-attn-3**: direct wheel URLs tied to specific torch+CUDA combinations. Listed under `no-build-isolation-package`.

## Common Commands

```bash
# Initial setup (transformers==5.9.0 via the default dependency group)
uv sync --extra gpu --dev

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
4. **uv version changes require Docker rebuilds** — update Dockerfiles and release new images. The Dockerfile uv pin must stay inside the `required-version` range in `pyproject.toml`.
5. **`override-dependencies` markers are load-bearing** — the `extra == 'gpu'` guards prevent uv from downloading wrong torch variants.
6. **`transformers==5.9.0` is the only supported version** — pinned via the `transformers-stable` default dependency group. New code targets v5 APIs (FSDP2 + patchgen-generated modeling) only.
