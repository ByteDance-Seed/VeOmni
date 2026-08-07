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
│   ├── gpu          NVIDIA x86_64 / aarch64 (glibc 2.34+) — full superset:
│   │                  torch 2.11.0+cu130 + cu130 nvidia stack + cuda-python
│   │                  + FA2 on x86_64 (cp311/cp312 wheels)
│   │                  + FA3 / FlashMLA wheels on both architectures
│   │                  + FA4 (PyPI) / FlashQLA (source-built from git)
│   │                  + liger-kernel + FLA + quack + TileLang/TileKernels + DLPack ext
│   │                  + diffusers / av / librosa / soundfile / ftfy / peft
│   │                  + megatron-energon (optional dataset format)
│   ├── npu          Ascend NPU x86_64 — full superset, minus CUDA-only kernels:
│   │                  torch 2.10.0+cpu + torch-npu 2.10.0
│   │                  + diffusers / av / audio / video / peft / megatron-energon
│   ├── npu_aarch64  Ascend NPU aarch64 — full superset except torchcodec:
│   │                  torch 2.10.0 + torch-npu 2.10.0 + the same data/model deps
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
│                            FA2 cp311/cp312, FA3 sm90 abi3, FlashMLA);
│                            git source (flash-qla)
└── uv.lock                  Lockfile (committed, used by Docker --locked)
```

## Hardware Extras (Mutually Exclusive)

`gpu` / `npu` / `npu_aarch64` are declared as conflicts. trl is not included
— VeOmni's DPO trainer is from-scratch.

```bash
uv sync --extra gpu --dev           # NVIDIA GPU
uv sync --extra npu --dev           # Ascend NPU x86
uv sync --extra npu_aarch64 --dev   # Ascend NPU ARM
```

A fresh `--extra gpu` installs architecture-specific torch, torchcodec, AV,
FA3, and FlashMLA wheels. FA2 is installed from prebuilt wheels on x86_64 and
omitted on aarch64. FA4 is a pure-Python wheel; only FlashQLA builds from git.
The aarch64 FA3 wheel requires glibc 2.34 or newer. uv caches built wheels
under `~/.cache/uv`.

The `npu` and `npu_aarch64` extras both install the complete Ascend software
stack and multimodal dependencies. Only `npu_aarch64` omits `torchcodec`
because no compatible aarch64 wheel is available; build it from source when
video decoding is required.

## Transformers Version

`transformers==5.9.0` is pinned by the `transformers-stable` group (in
`default-groups`). Kept out of `[project.dependencies]` so pip users are not
forced into a specific 5.x patch.

## torch Source Pinning

- **GPU**: direct cp311/cp312 wheel URLs for x86_64 and aarch64 (not the
  pytorch index) — avoids uv resolving cu128_full wheels that drop nvidia-* deps.
- **NPU**: pytorch index (`https://download.pytorch.org/whl/`).

## Attention Kernels

| Package | Source | Notes |
|---|---|---|
| `flash-attn` (FA2) | cp311 wheel (v0.0.3) + cp312 wheel (v0.0.5), Luosuu cu130/torch2.11/sm80-100 | x86_64 only; omitted on aarch64 |
| `flash-attn-3` (Hopper) | cp310-abi3 Luosuu wheel on x86_64; cp39-abi3 PyTorch cu130 wheel on aarch64 | abi3 covers supported Python versions; aarch64 requires glibc 2.34+ |
| `flash-mla` | cp311/cp312 Luosuu cu130/torch2.11/sm90a+sm100f wheels | architecture-specific x86_64/aarch64 wheels |
| `flash-attn-4` (cute) | PyPI `4.0.0b16` | pure-Python wheel |
| `flash-qla` | git: QwenLM/FlashQLA | source-built; uv overrides its TileLang 0.1.8 metadata pin |
| `tile-kernels` | PyPI `1.0.0` | DeepSeek V4 mHC forward/backward; requires TileLang 0.1.9 and SM90+ |

Two pyproject knobs make the remaining FlashQLA source build succeed:

1. **`[[tool.uv.dependency-metadata]]`** with `version` for `flash-qla`.
   It has no `pyproject.toml`; without static metadata uv runs its setup.py
   on a fresh venv and crashes with `ModuleNotFoundError: No module named
   'setuptools'`. The static `requires-dist` mirrors flash-qla's own
   install_requires (`torch`, `tilelang==0.1.8`, `apache-tvm-ffi==0.1.9`)
   — `flash_qla/__init__.py` top-level imports `tilelang`, so they have to
   ship alongside, even though the `flash_qla` kernel itself only binds on
   sm90 (gated by `KernelSpec(min_compute_capability=90)`).

   The GPU extra also installs `tile-kernels==1.0.0`, which requires
   `tilelang>=0.1.9`. `[tool.uv].override-dependencies` therefore pins the
   shared GPU environment to `tilelang==0.1.9`; DeepSeek V4 TileLang and
   TileKernels tests cover that resolved combination.

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
3. **FA2/FA3/FlashMLA wheels are pinned to torch 2.11 cu130 cp311/cp312/abi3.**
   Bumping torch / Python / cuda requires matching PyTorch/Luosuu releases.
4. **uv bumps require Docker rebuilds**; concrete pins must stay in range.
5. **`override-dependencies` `extra == '...'` markers are load-bearing.**
6. **`transformers==5.9.0` is the only supported version.** New code targets
   v5 + FSDP2 + patchgen-generated modeling.
