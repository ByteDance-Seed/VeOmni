# VeOmni Development Guide

> Instructions for AI coding agents working on this repository.

**VeOmni** is a modular distributed training framework for multi-modality models (text, vision, audio, diffusion, omni) across various accelerators (GPUs, NPUs). Developed by ByteDance Seed Team.

- Homepage: https://github.com/ByteDance-Seed/VeOmni
- Python: `>=3.11, <3.12`
- Package: `veomni`

**Language**: Match user's language (Chinese or English).

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
- **Planning Discipline**: Complex tasks (multi-file, >30 min) -> TodoWrite. Simple tasks -> just do them.
- **Cross-modality Awareness**: Changes in shared code (`BaseTrainer`, `data_collator`, `distributed/`) affect all modalities.
- **No Patchgen Edits**: Never edit files under `veomni/models/transformers/*/generated/`.

---

## Setup

```bash
uv sync --extra gpu --extra audio --dev
source .venv/bin/activate
```

Always activate `.venv/` before running any commands.

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
```

---

## PR Guidelines

Title: `[{modules}] {type}: {description}`

- `{modules}`: `misc`, `ci`, `config`, `docs`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `perf`, `ops`, `parallel`
- `{type}`: `feat`, `fix`, `refactor`, `chore`, `test`
- Breaking: prepend `[BREAKING]`

---

## Commit Flow

1. Complete and verify the change.
2. Run `veomni-review` skill (subagent code review).
3. **safe** -> commit. **risky** -> report to user, wait for approval.
4. Each fix -> immediate commit. Do not batch unrelated changes.
5. Run `make quality` before every commit.
6. **Commit messages must NOT mention Claude/AI/Co-Authored-By.**

---

## Skill Dispatch

Read the skill file before starting.

| Task | Skill | Path |
|------|-------|------|
| New feature | `veomni-feature` | `.agents/skills/veomni-feature.md` |
| Simple bug fix | `veomni-bugfix` | `.agents/skills/veomni-bugfix.md` |
| Complex debugging | `veomni-debug` | `.agents/skills/veomni-debug/SKILL.md` |
| Refactoring | `veomni-refactor` | `.agents/skills/veomni-refactor.md` |
| Code review (pre-commit) | `veomni-review` | `.agents/skills/veomni-review/SKILL.md` |
| Verify conclusions | `veomni-verify` | `.agents/skills/veomni-verify/SKILL.md` |
| Add new model | `veomni-new-model` | `.agents/skills/veomni-new-model/SKILL.md` |
| Add new op/kernel | `veomni-new-op` | `.agents/skills/veomni-new-op/SKILL.md` |
| Run tests | `veomni-run-test` | `.agents/skills/veomni-run-test/SKILL.md` |
| Update dependencies (uv) | `veomni-uv-update` | `.agents/skills/veomni-uv-update/SKILL.md` |
| Post-compaction recovery | `veomni-housekeeping` | `.agents/skills/veomni-housekeeping/SKILL.md` |

### Quick Decision Guide

- **"Add support for model X"** -> `veomni-new-model`
- **"Add a new kernel / fused op"** -> `veomni-new-op`
- **"Fix this error"** (clear cause) -> `veomni-bugfix`
- **"Training is hanging / wrong results"** (unclear cause) -> `veomni-debug`
- **"Add a new capability"** -> `veomni-feature`
- **"Clean up / reorganize"** -> `veomni-refactor`
- **"Update package X" / "bump uv" / "upgrade torch"** -> `veomni-uv-update`
- **"Is this conclusion correct?"** -> `veomni-verify`
