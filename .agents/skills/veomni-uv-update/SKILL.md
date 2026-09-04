---
name: veomni-uv-update
description: "Use this skill when updating dependencies managed by uv: bumping a package version, upgrading the uv tool itself, updating torch/CUDA stack, switching transformers version, or regenerating the lockfile. Trigger: 'update dependency', 'bump version', 'upgrade uv', 'update torch', 'update lockfile', 'uv sync fails'."
---

## Before You Start

Read `.agents/knowledge/uv.md` for the full dependency architecture. The key things that make VeOmni's uv setup non-trivial:

- `[tool.uv].required-version` is a **range**; the concrete uv pins live
  elsewhere and must stay inside it
- every Dockerfile is standalone and hand-maintained; there is no generator or
  matrix, so a version bump has to be applied file by file
- torch uses **direct wheel URLs** (not just version bumps)
- only three extras: `gpu` / `npu` / `npu_aarch64`, mutually exclusive,
  each a complete superset

`pyproject.toml` is the source of truth for every version claim below. Read the
relevant block before editing — this file describes *where* things live, not
which versions are current.

## Scenario 1: Update uv Version

`pyproject.toml` -> `[tool.uv]` -> `required-version` is a **range**
(e.g. `">=0.9.8,<0.13"`). Docker and CI install a **concrete** pin and run with
`--locked` / `--frozen`. Every concrete pin must stay inside the range.

1. **Every Dockerfile that pins uv, one by one.** There is no generator; the
   pin lives in a `COPY --from=ghcr.io/astral-sh/uv:X.Y.Z` line, and only the
   uv-based images have one (the pip-based ascend `*.arm` / `*_a3` variants do
   not). Enumerate rather than assume:

   ```bash
   grep -rn "astral-sh/uv" docker/
   ```

   Update every hit, and keep them on the same version — a per-image drift is
   a debugging trap, not a feature.
2. `.github/workflows/check_patchgen.yml` -> `astral-sh/setup-uv` `version:`.
   This job runs outside the container image, so an unpinned uv would float
   above the range ceiling.
3. `pyproject.toml` -> `required-version` — only widen/move the range when the
   new pin falls outside it.

Then regenerate the lockfile:

```bash
uv lock
uv sync --extra gpu --dev
```

Verify the lockfile diff is reasonable (`git diff uv.lock` — should only show version changes, not wholesale rewrites).

## Scenario 2: Update a Regular Dependency

1. Edit version constraint in `pyproject.toml` under `[project.dependencies]` or the relevant `[project.optional-dependencies]` extra.
2. Regenerate lockfile and sync:

```bash
uv lock
uv sync --extra gpu --dev
```

3. Run tests: `pytest tests/`
4. Commit both `pyproject.toml` and `uv.lock` together.

## Scenario 3: Update torch / CUDA Stack

This is the most complex update. torch versions are pinned in **multiple places**:

**For GPU (`gpu` extra):**
- `pyproject.toml` -> `[project.optional-dependencies]` -> `gpu` list
- `pyproject.toml` -> `[tool.uv]` -> `override-dependencies` (the `extra == 'gpu'` entries)
- `pyproject.toml` -> `[tool.uv.sources]` -> `torch` (direct wheel URL — must update to matching wheel)
- Related packages that must move together: `torchvision`, `torchaudio`,
  `torchcodec`, plus the `nvidia-*` runtime pins in the `gpu` extra. Grep the
  `gpu` block rather than trusting this list — it grows.

**For NPU (`npu` / `npu_aarch64` extras):**
- Same pattern but with `+cpu` suffix or no suffix

**Steps:**
1. Identify the target torch version and matching wheel URLs from https://download.pytorch.org/whl/
2. Update all pinned versions in `pyproject.toml` (extras, overrides, sources)
3. Check attention-kernel compatibility. Three groups behave differently —
   confirm each against `[tool.uv.sources]` before editing:
   - **Prebuilt wheel URLs** (`flash-attn` cp311/cp312 x86_64-only,
     `flash-attn-3` abi3, `flash-mla`): pinned to torch+CUDA+ABI-specific
     wheels. A torch / Python / CUDA bump requires a matching upstream release
     — see https://github.com/Luosuu/flash-attention3-wheels/releases.
   - **PyPI releases** (`flash-attn-4`, `flash-qla`): plain version pins in the
     `gpu` extra. `flash-qla` is a pure-Python wheel whose static metadata
     already matches the `tilelang` / `apache-tvm-ffi` pins, so it needs no
     source build and no `dependency-metadata` override. `tilelang` is pinned
     in `override-dependencies` because `tile-kernels` and `flash-qla` must
     agree on one version — bump them as a set.
   - **Source-built git pins** (`magi-attention`, `create-block-mask-cuda`,
     `flash-attn-cute`, `magi-to-hstu-cuda`): each needs a
     `[[tool.uv.dependency-metadata]]` block (upstream declares no usable
     metadata) plus an `extra-build-dependencies` entry, and an
     `extra-build-variables` entry where the build needs `MAX_JOBS` /
     compute-capability flags (all but `flash-attn-cute` today).
     A torch ABI bump may require bumping the git revs. These are
     SM90+ only and the GPU CI job installs them with `--no-install-package`
     exclusions on the SM89 L20 runners.
4. Update `torchcodec` version if needed (compatibility note in pyproject.toml)
5. Regenerate lockfile:

```bash
uv lock
uv sync --extra gpu --dev
```

6. Run tests: `pytest tests/`
7. If the torch version changed, walk the Dockerfiles. Seven of them pin torch
   directly — `docker/rocm/Dockerfile.ROCm7.14` a ROCm build, and the ascend
   `*_torch_npu*` images a `torch-npu==X` matched to it by `fla_npu`'s
   `check_npu_env`. The rest inherit torch from their base image
   (`docker/cuda/Dockerfile.cu130` from the NGC PyTorch base), so there is no
   single knob. Match `-npu` too, or you will find one pin out of seven:

   ```bash
   grep -rnE "torch(-npu)?==" docker/
   ```

## Scenario 4: Update transformers Version

transformers is pinned by the `transformers-stable` dependency group
(`pyproject.toml` -> `[dependency-groups] transformers-stable`), which is
listed in `[tool.uv] default-groups` so `uv sync` installs it automatically.

**Bump within v5** (e.g. 5.2.0 → 5.3.0):
1. Edit the pinned version in `[dependency-groups] transformers-stable`.
2. Regenerate lockfile and sync:

```bash
uv lock
uv sync --extra gpu --dev
```

3. Check for API breakage and adjust `veomni/` accordingly. Forward-looking
   guards may be expressed with
   `is_transformers_version_greater_or_equal_to()` from
   `veomni/utils/import_utils.py`.
4. Run tests: `pytest tests/models/ tests/e2e/`
5. Regenerate model patches: `make patchgen` (with the target transformers installed)

## Scenario 5: Regenerate Lockfile Only

When `uv.lock` is out of sync or corrupt:

```bash
uv lock
uv sync --extra gpu --dev
```

If `uv lock` fails due to version conflicts, check:
- `[tool.uv]` -> `conflicts` declarations
- `override-dependencies` markers
- Direct wheel URL availability

## Common Pitfalls

- **Bumping one Dockerfile and calling it done**: there are a dozen-plus standalone Dockerfiles under `docker/` and no generator to fan a change out. `grep -rn` for the pin you are moving and update every hit.
- **Partial torch updates**: updating `torch` but not `torchvision`/`torchaudio`/`torchcodec` to matching versions causes import errors.
- **flash-attn wheel mismatch**: flash-attn wheels are built for specific torch+CUDA combinations. A torch version bump requires finding or building new wheels.
- **Committing only pyproject.toml**: always commit `uv.lock` together. Docker builds use `--locked` which requires the lockfile to match.
- **override-dependencies markers**: the `extra == 'gpu'` markers in overrides are critical. Removing them causes uv to download wrong torch variants from PyPI.
- **Assuming build isolation is disabled**: there is no `no-build-isolation-package` block any more. Source builds instead get their toolchain from `[tool.uv.extra-build-dependencies]` (uv venvs are not seeded), and `torch` is passed with `match-runtime = true` where the extension links against it. If a source build fails on a missing `setuptools`/`torch`, add it there rather than reaching for `--no-build-isolation`.
- **Overlay reinstall**: an exact `uv sync` removes the MagiAttention SM90 CUTLASS overlay installed by `scripts/kernel/install_magi_sm90.sh`. Reinstall it afterwards (see constraints, "Environment Reproducibility").
