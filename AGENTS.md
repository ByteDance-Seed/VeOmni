# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Instructions for AI coding agents working on this repository.

**VeOmni** is a modular distributed training framework for multi-modality models (text, vision, audio, diffusion, omni) across various accelerators (GPUs, NPUs). Developed by ByteDance Seed Team.

- Homepage: https://github.com/ByteDance-Seed/VeOmni
- Python: `>=3.11, <3.12`
- Package: `veomni`

**Language**: Match user's language (English).

## Context Loading

On session start, read the following:
- `.agents/knowledge/constraints.md` — hard constraints to check before any code change
- `.agents/knowledge/architecture.md` — module map, trainer hierarchy, data flow
- `.agents/knowledge/uv.md` — dependency management architecture (uv, extras, lockfile)

---

## Core Principles

- **Challenge First, Execute Second**: Spot logic flaws or simpler alternatives? Raise concerns before executing.
- **Explain, Don't Assume**: Explain **why** (motivation, tradeoffs), not just what. Cite files and line numbers.
- **Ask When Stuck**: 3+ approaches fail? Stop, summarize, ask user. No hacks.
- **Search Before You Act**: On unexpected behavior, search codebase + check constraints + review `git log` before attempting fixes.
- **Planning Discipline**: Complex tasks (multi-file, >30 min) -> TodoWrite. Plan must state which skills will be used (e.g. `/veomni-develop` + `/veomni-review`). Simple tasks -> just do them.
- **Cross-modality Awareness**: Changes in shared code (`BaseTrainer`, `data_collator`, `distributed/`) affect all modalities.
- **No Patchgen Edits**: Never edit files under `veomni/models/transformers/*/generated/`.

---

## Setup

```bash
# Default (transformers 5.2.0)
uv sync --extra gpu --extra audio --dev
source .venv/bin/activate

# Legacy escape hatch for transformers 4.57.3 (sunset path):
uv sync --no-group transformers-stable --extra transformers-v4-legacy --extra gpu --extra audio --dev
```

Always activate `.venv/` before running any commands. New code must target transformers v5 and FSDP2. See `.agents/knowledge/constraints.md` for details.

---

## Development Commands

```bash
source .venv/bin/activate
make style          # ruff fix + format
make quality        # ruff check (CI gate)
make commit         # style + quality
make patchgen       # regenerate model patches
pytest tests/       # all tests
pytest tests/<mod>/ # specific module
pytest tests/models/test_models_patch.py -k <model_name>  # single model patch test
pytest --collect-only -k <model_name>                     # list tests for a model
python3 scripts/ci/check_doc_task_paths.py                # validate doc task paths after renaming tasks/
```

When moving/renaming scripts under `tasks/`, search `docs/` for old paths and update them.

---

## Testing by Change Area

| Changed module | Test command |
|---|---|
| `veomni/models/` | `pytest tests/models/` |
| `veomni/data/` | `pytest tests/data/` |
| `veomni/ops/` | `pytest tests/ops/` |
| `veomni/distributed/` | `pytest tests/parallel/` |
| `veomni/checkpoint/` | `pytest tests/checkpoints/` |
| `veomni/utils/` | `pytest tests/utils/` |
| `veomni/trainer/` | `pytest tests/e2e/` |

Distributed tests (`tests/parallel/`, `tests/e2e/`, `tests/distributed/`) require multiple GPUs and use `torchrun` or `tests/tools/launch_utils.py`. See `docs/testing.md` for the full test catalog and new-model onboarding checklist.

---

## DeepSpeed ZeRO

DeepSpeed is a third parallelism backend alongside `ddp` and `fsdp2`. Key differences from FSDP2:

**YAML config skeleton:**

```yaml
train:
  init_device: cpu          # required (FSDP2 uses meta)
  accelerator:
    fsdp_config:
      fsdp_mode: deepspeed
  checkpoint:
    manager: deepspeed      # required (dcp only works with ddp/fsdp2)
  deepspeed:
    zero_stage: 3           # 1 | 2 | 3 (default 3)
    offload_optimizer: cpu  # "cpu" | "nvme" | null
    offload_param: cpu      # "cpu" | "nvme" | null  — zero_stage 3 only
    nvme_path: /mnt/nvme    # required when offload target is "nvme"
    overlap_comm: true
    contiguous_gradients: true
    config_path: ""         # set to load a raw JSON config (overrides all fields above)
```

**Hard constraints** (see `.agents/knowledge/constraints.md` §DeepSpeed for full detail):

- `init_device: cpu` is mandatory — the argument validator raises if `fsdp_mode: deepspeed` is paired with `init_device: meta`.
- `checkpoint.manager: deepspeed` is mandatory — `manager: dcp` rejects `dist_backend=deepspeed`.
- Optimizer and lr_scheduler are built *before* `init_deepspeed_engine()`, opposite to FSDP2 order (`veomni/trainer/base.py:196`).
- `engine.step()` handles grad clip + optimizer.step + lr_scheduler.step + zero_grad atomically; never call these separately.
- `offload_param` requires `zero_stage: 3`; the `DeepSpeedConfig` validator enforces this.
- SP (Ulysses) and EP are FSDP2-only — `ulysses_size > 1` or `ep_size > 1` with `fsdp_mode: deepspeed` is unsupported.
- `config_path` (a raw DeepSpeed JSON file) overrides all `train.deepspeed.*` fields; only `train_batch_size` and `gradient_accumulation_steps` are patched in from `TrainingArguments`.

**Key files:**

| File | Role |
|---|---|
| `veomni/distributed/deepspeed_init.py` | `build_ds_config()`, `init_deepspeed_engine()` |
| `veomni/checkpoint/ds_checkpointer.py` | `DeepSpeedCheckpointer` (engine.save/load_checkpoint) |
| `veomni/arguments/arguments_types.py` | `DeepSpeedConfig` dataclass (`train.deepspeed.*`) |
| `veomni/trainer/base.py:315` | `_is_deepspeed_mode` property; build-order and step logic |

---

## PR Guidelines

Title: `[{modules}] {type}: {description}`

**Allowed modules**: `misc`, `ci`, `config`, `docs`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `perf`, `ops`, `parallel`, `docker`, `trainer`, `agent`, `lora`

**Allowed types**: `feat`, `fix`, `refactor`, `chore`, `test`

Multiple modules: `[parallel, model] feat: ...`  
Breaking change: `[BREAKING][model] feat: ...`

---

## Commit Flow

1. Complete and verify the change.
2. Update related documentation: `docs/`, `README.md`, `.agents/knowledge/`, config examples — if the change introduces, modifies, or removes any API, config field, or workflow.
3. Run `/veomni-review` skill (subagent code review).
4. **safe** -> commit. **risky** -> report to user, wait for approval.
5. Each fix -> immediate commit. Do not batch unrelated changes.
6. Run `make quality` before every commit.
7. **Commit messages must NOT mention Claude/AI/Co-Authored-By.**
8. **Skill gap check**: If the task didn't match any existing skill, briefly assess after completion: Was this a one-off, or a repeatable pattern? If repeatable, suggest creating a new skill to the user.

---

## Skills

Skills follow the [Agent Skills](https://agentskills.io) open standard. Each skill is a folder in `.agents/skills/<name>/` containing a `SKILL.md` with YAML frontmatter (`name`, `description`). Skills are auto-discovered by compatible agents (Cursor, Claude Code, Codex, etc.) and can also be invoked manually with `/skill-name` in chat.

| Task | Skill |
|------|-------|
| Feature / refactoring | `/veomni-develop` |
| Bug fix / debugging | `/veomni-debug` |
| Code review (pre-commit) | `/veomni-review` |
| Add new model | `/veomni-new-model` |
| Migrate existing model to transformers v5 | `/veomni-migrate-transformers-v5` |
| Add new op/kernel | `/veomni-new-op` |
| Update dependencies (uv) | `/veomni-uv-update` |
| Performance profiling | `/veomni-profile` |

### Quick Decision Guide

- **"Add support for model X"** → `/veomni-new-model`
- **"Migrate X to transformers v5" / "port X to patchgen" / "convert monkey patch to generated modeling"** → `/veomni-migrate-transformers-v5`
- **"Add a new kernel / fused op"** → `/veomni-new-op`
- **"Fix this error" / "training hangs" / "wrong results"** → `/veomni-debug`
- **"Add a new capability" / "refactor" / "clean up"** → `/veomni-develop`
- **"Update package X" / "bump uv" / "upgrade torch"** → `/veomni-uv-update`
- **"Analyze this trace" / "why is training slow" / "profile" / "MFU"** → `/veomni-profile`
