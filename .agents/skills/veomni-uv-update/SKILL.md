---
name: veomni-uv-update
description: "Update VeOmni dependencies managed by uv: bump packages, upgrade uv itself, update torch/CUDA/NPU stacks, bump transformers, or regenerate uv.lock. Trigger: 'update dependency', 'bump version', 'upgrade uv', 'update torch', 'update lockfile', 'uv sync fails'."
---

# uv Dependency Architecture & Update Playbook

Always inspect `pyproject.toml` as the source of truth for current exact versions before editing.

## Architecture & Hardware Extras

- **`[project.dependencies]`**: core deps (`transformers` excluded so pip users aren't pinned to one patch; pinned instead in `[dependency-groups].transformers-stable` via `default-groups`).
- **Mutually exclusive hardware extras** (`[tool.uv].conflicts`):
  - `gpu`: NVIDIA x86_64/aarch64 superset (`torch` + cu130 stack, FA2 on x86_64, FA3/FlashMLA/DeepGEMM wheels, FA4/FlashQLA/TileKernels, `liger-kernel`, FLA, `quack`, diffusers/av/audio/peft/energon).
  - `npu` / `npu_aarch64`: Ascend NPU supersets (`torch` + `torch-npu` + multimodal deps; `npu_aarch64` omits `torchcodec`).
- **Optional GPU add-on extras** (combine with `--extra gpu`):
  - `magi`: SM100+ source-built MagiAttention + 3 companion CUDA extensions (conflicts with NPU extras; omitted from default GPU CI). Uses `[[tool.uv.dependency-metadata]]` (`requires-dist = []`), `[tool.uv.extra-build-dependencies]` (`setuptools`/`wheel`/`packaging`/`ninja` + `torch` with `match-runtime = true`), and `[tool.uv.extra-build-variables]` rather than `--no-build-isolation`.
  - `float8`: `torchao==0.18.0` (exact pin checked at runtime by `veomni/ops/torchao.py`; incompatible with `peft==0.18.1` torchao interop, omitted from default GPU CI).
- **Optional NPU add-on extra** (combine with `--extra npu` or `--extra npu_aarch64`):
  - `hyper`: HyperParallel Ascend backend (pinned Git source in `[tool.uv.sources]`; see `docs/key_features/hyper_parallel.md`).
- **Direct wheel URLs (`[tool.uv.sources]`)**: GPU `torch` uses direct cp311/cp312 wheel URLs (not the PyTorch index) to avoid `cu128_full` wheels dropping `nvidia-*` deps; `deep-gemm` pins 4 arch/Python wheel URLs (requires glibc >=2.38, NVCC + C++20 JIT; never add to CPU/NPU).
- **Load-bearing overrides**: `[tool.uv].override-dependencies` requires `extra == '...'` markers (removing them pulls wrong PyTorch wheels) and pins `tilelang` shared by `flash-qla` and `tile-kernels` (bump as a set).

## Common Commands

```bash
uv sync --extra gpu --dev                          # local dev (cp311/cp312)
uv sync --extra gpu --extra magi --dev             # + MagiAttention (SM100+)
uv sync --extra gpu --extra float8 --dev           # + torchao (float8 / FP4 QAT)
uv sync --extra npu --dev                          # Ascend NPU x86_64 (or npu_aarch64)
uv sync --extra npu --extra hyper --dev            # + HyperParallel (Ascend NPU)
uv lock && uv sync --extra gpu --dev               # after pyproject.toml edits
uv sync --locked --all-packages --extra gpu --dev  # Docker / CI
```

## Update Procedures

1. **Update `uv` version**: Dockerfiles are standalone and hand-maintained (`*.arm`/`*_a3` pip-based ascend images have no uv pin). Enumerate and update every hit to the same `X.Y.Z` pin, widening `[tool.uv].required-version` only if outside range:
   ```bash
   grep -rn "astral-sh/uv" docker/
   ```
   Then run `uv lock && uv sync --extra gpu --dev` and inspect `git diff uv.lock`.

2. **Update a regular package**: edit `pyproject.toml`, run `uv lock && uv sync --extra gpu --dev`, run relevant tests, and commit `pyproject.toml` + `uv.lock` together.

3. **Update `torch` / CUDA / NPU stack**:
   - Update all pinned sites in `pyproject.toml`: `[project.optional-dependencies]` (incl. `torchvision`/`torchaudio`/`torchcodec`/`nvidia-*`), `[tool.uv].override-dependencies` (preserve `extra == '...'` markers), and `[tool.uv.sources]` direct wheel URLs.
   - Verify ABI-pinned wheels (`flash-attn`, `flash-attn-3`, `flash-mla`, `deep-gemm`), PyPI set (`flash-qla` + `tile-kernels` + `tilelang`), and `--extra magi` git revs.
   - Enumerate and update every Dockerfile pinning torch directly:
     ```bash
     grep -rnE "torch(-npu)?==" docker/
     ```
   - Run `uv lock && uv sync --extra gpu --dev` and test.

4. **Update `transformers`**: edit `[dependency-groups].transformers-stable`, run `uv lock && uv sync --extra gpu --dev`, audit `veomni/models/transformers/*/modeling_*.py` (updating `# Base-Transformers-Version:` headers when rebased), and run `pytest tests/models/ tests/e2e/`.

5. **Regenerate lockfile only**: run `uv lock && uv sync --extra gpu --dev`; on conflict, inspect `[tool.uv].conflicts`, `override-dependencies` markers, and wheel URL availability.
