# AGENTS.md

> Instructions for AI coding agents working on this repository.

## Project Overview

**VeOmni** is a versatile, modular framework for scaling any modality model training (language, vision, audio, diffusion, omni-models) across various accelerators (GPUs, NPUs). Developed by ByteDance Seed Team.

- Homepage: https://github.com/ByteDance-Seed/VeOmni
- Python: `>=3.11, <3.12`
- Package: `veomni`

### Key Principles

- **Trainer-free**: Linear training scripts without rigid trainer classes
- **Flexibility & Modularity**: Decouple components for custom implementations
- **PyTorch native**: Leverage native PyTorch functions

---

## Repository Structure

```
VeOmni/
├── veomni/           # Main package
│   ├── arguments/    # CLI argument definitions
│   ├── checkpoint/   # DCP checkpoint management
│   ├── data/         # Data handling, collators, dynamic batching
│   ├── distributed/  # FSDP, FSDP2, Sequence Parallel, MOE Expert Parallel
│   ├── models/       # Auto model loader (HuggingFace compatible)
│   ├── ops/          # Kernel optimizations (flash-attn, fused kernels)
│   ├── optim/        # Optimizers and LR schedulers
│   ├── patchgen/     # Patch generation utilities
│   ├── schedulers/   # LR scheduler implementations
│   ├── trainer/      # Trainer utilities
│   └── utils/        # Logging, device management, misc helpers
├── configs/          # Model configs
│   ├── dit/          # Diffusion transformer configs
│   ├── model_configs/# Base model configurations
│   ├── multimodal/   # Vision-language model configs
│   └── text/         # Language model configs
├── tasks/            # Training entry points
│   ├── train_text.py         # Text/LLM training
│   ├── train_text_rl.py      # Text RLHF training
│   ├── train_vlm.py          # Vision-Language model training
│   ├── train_vlm_rl.py       # VLM RLHF training
│   ├── dit/                  # Diffusion model tasks
│   ├── infer/                # Inference tasks
│   └── omni/                 # Omni-model tasks
├── docs/             # Documentation (MkDocs)
├── scripts/          # Utility scripts
└── tests/            # Test suite
    ├── checkpoints/  # Checkpoint tests
    ├── data/         # Data pipeline tests
    ├── e2e/          # End-to-end tests
    ├── models/       # Model tests
    ├── ops/          # Ops/kernel tests
    └── parallel/     # Distributed/parallel tests
```

---

## Setup

```bash
uv sync --extra gpu --extra audio --dev
```

The virtual environment is created at `.venv/`. **Always use binaries from `.venv/bin/` directly** — do not `source .venv/bin/activate`.

---

## Development Commands

```bash
# Format and lint (run before committing)
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .

# Full pre-commit check
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files --show-diff-on-failure --color=always

# Makefile shortcuts (internally use .venv binaries)
make style    # ruff format
make quality  # ruff check
make commit   # style + quality
```

---

## Testing

CI workflows are defined in [.github/workflows/](.github/workflows/):

- `gpu_unit_tests.yml` — GPU unit tests
- `npu_unit_tests.yml` — NPU (Ascend) unit tests
- `gpu_e2e_test.yml` — End-to-end GPU tests
- `check_pr_title.yml` — Enforces PR title format

Run tests locally using `.venv/bin/pytest`:

```bash
# All unit tests
.venv/bin/pytest tests/

# Specific module
.venv/bin/pytest tests/models/
.venv/bin/pytest tests/parallel/
```

---

## Code Style

- **Formatter**: `ruff format` (via `make style`)
- **Linter**: `ruff check` (via `make quality`)
- All code comments and docstrings must be in **English**
- Function signatures and configuration default values should be centralized (config files, constants modules, or main entry points) rather than scattered throughout the codebase

---

## PR Guidelines

### Title Format

```
[{modules}] {type}: {description}
```

- `{modules}`: `misc`, `ci`, `config`, `docs`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `perf`, `ops`, `parallel`
  - Multiple modules: `[ci, data, model]`
- `{type}`: `feat`, `fix`, `refactor`, `chore`, `test`
- Breaking changes: prepend `[BREAKING]`
  - Example: `[BREAKING][parallel, model] feat: dynamic batching`

### PR Description Template

```markdown
### What does this PR do?

> Concise overview of the change. Reference related issues/PRs.

### Checklist Before Starting

- [ ] Search for similar PRs. Paste at least one query link here: ...
- [ ] PR title follows `[{modules}] {type}: {description}` format

### Test

> Validation results (training curves, eval metrics) for changes not covered by CI.

### API and Usage Example

> Show API changes and usage examples if applicable.

### Design & Code Changes

> High-level design description and specific change list.

### Checklist Before Submitting

- [ ] Read the [Contribute Guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md)
- [ ] Applied pre-commit checks
- [ ] Added/updated documentation
- [ ] Added tests to CI workflow (or explained why not feasible)
```

Use `git diff main` to view the changes in the current branch.

---

## Supported Models

| Category | Models |
|---|---|
| LLMs | DeepSeek, Llama 3, Qwen 2/3 (up to 72B/671B) |
| Vision-Language | Qwen3-VL, QVQ (2B–72B) |
| MoE | Qwen3-MoE, Qwen3-VL MoE |
| Diffusion | Wan 2.1-I2V (up to 14B) |

---

## Hardware Support

- **GPU**: CUDA 12.8 (NVIDIA)
- **NPU**: Ascend (Huawei)

---

## Key Features

- FSDP and FSDP2 distributed training
- Sequence Parallelism (DeepSpeed Ulysses)
- Expert Parallelism for MoE models
- HuggingFace Transformers compatibility (`transformers==4.57.3` stable)
- DCP (Distributed Checkpoint) management
