---
name: veomni-uv-update
description: "Use this skill when updating dependencies managed by uv: bumping a package version, upgrading the uv tool itself, updating torch/CUDA stack, switching transformers version, or regenerating the lockfile. Trigger: 'update dependency', 'bump version', 'upgrade uv', 'update torch', 'update lockfile', 'uv sync fails'."
---

## Before You Start

Read `.agents/knowledge/uv.md` for the full dependency architecture. The key things that make VeOmni's uv setup non-trivial:

- uv version is pinned in **three places** (must update together)
- torch uses **direct wheel URLs** (not just version bumps)
- hardware extras (`gpu`, `npu`, `npu_aarch64`) are **mutually conflicting**

## Scenario 1: Update uv Version

uv is pinned to a specific version. Update **all three locations** together:

1. `pyproject.toml` -> `[tool.uv]` -> `required-version = "==X.Y.Z"`
2. `docker/cuda/Dockerfile.cu129` -> `COPY --from=ghcr.io/astral-sh/uv:X.Y.Z`
3. `docker/ascend/Dockerfile.ascend_*` -> same pattern (if present)

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
- Related packages: `torchvision`, `torchaudio`, `torchcodec`, `nvidia-cudnn-cu12`

**For NPU (`npu` / `npu_aarch64` extras):**
- Same pattern but with `+cpu` suffix or no suffix

**Steps:**
1. Identify the target torch version and matching wheel URLs from https://download.pytorch.org/whl/
2. Update all pinned versions in `pyproject.toml` (extras, overrides, sources)
3. Check `flash-attn` / `flash-attn-3` wheel compatibility — these are tied to specific torch versions via direct URLs in `[tool.uv.sources]`
4. Update `torchcodec` version if needed (compatibility note in pyproject.toml)
5. Regenerate lockfile:

```bash
uv lock
uv sync --extra gpu --dev
```

6. Run tests: `pytest tests/`
7. Update Docker images if torch version changed

## Scenario 4: Update transformers Version

transformers uses a **dual-track** setup: default `4.57.3` (group `transformers-stable`) and experimental `5.2.0` (extra `transformers5-exp`), declared as conflicts in `[tool.uv.conflicts]`.

**Bump within a track** (e.g. 4.57.3 → 4.58.0, or 5.2.0 → 5.3.0):
1. Edit the pinned version in the relevant section of `pyproject.toml`:
   - Stable: `[dependency-groups]` -> `transformers-stable`
   - Experimental: `[project.optional-dependencies]` -> `transformers5-exp`
2. Regenerate lockfile and sync:

```bash
uv lock
# Stable:
uv sync --extra gpu --dev
# Or v5:
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --dev
```

3. Check for API breakage — key v4→v5 differences:
   - `AutoModelForVision2Seq` removed (use `AutoModelForImageTextToText`)
   - `no_init_weights` moved from `transformers.modeling_utils` to `transformers.initialization`
   - Model `__init__.py` version gates: `is_transformers_version_greater_or_equal_to("5.0.0")` chooses `generated/` (v5) vs upstream + `apply_*_patch()` (v4)
   - Some models (e.g. `qwen3_5`, `glm_moe_dsa`) only register on `>= 5.2.0`
4. Run tests: `pytest tests/models/ tests/e2e/`
5. Regenerate model patches if needed: `make patchgen` (with the target transformers installed)

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

- **Forgetting to update Docker**: uv version and torch version changes must be reflected in `docker/` Dockerfiles, otherwise CI builds will fail.
- **Partial torch updates**: updating `torch` but not `torchvision`/`torchaudio`/`torchcodec` to matching versions causes import errors.
- **flash-attn wheel mismatch**: flash-attn wheels are built for specific torch+CUDA combinations. A torch version bump requires finding or building new wheels.
- **Committing only pyproject.toml**: always commit `uv.lock` together. Docker builds use `--locked` which requires the lockfile to match.
- **override-dependencies markers**: the `extra == 'gpu'` markers in overrides are critical. Removing them causes uv to download wrong torch variants from PyPI.
- **no-build-isolation**: `flash-attn` and `flash-attn-3` are listed under `no-build-isolation-package`. They require torch to be installed first. If sync fails, try `uv sync` without these extras first, then add them.
