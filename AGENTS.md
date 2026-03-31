# VeOmni Development Guide

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
# Default (transformers 4.57.3)
uv sync --extra gpu --extra audio --dev
source .venv/bin/activate

# For new development targeting transformers v5:
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --extra audio --dev
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
```

---

## PR Guidelines

Title: `[{modules}] {type}: {description}`

- Allowed modules and types are defined in `.github/workflows/check_pr_title.yml` (the CI source of truth).
- Breaking: prepend `[BREAKING]`

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
| Add new op/kernel | `/veomni-new-op` |
| Update dependencies (uv) | `/veomni-uv-update` |

### Quick Decision Guide

- **"Add support for model X"** → `/veomni-new-model`
- **"Add a new kernel / fused op"** → `/veomni-new-op`
- **"Fix this error" / "training hangs" / "wrong results"** → `/veomni-debug`
- **"Add a new capability" / "refactor" / "clean up"** → `/veomni-develop`
- **"Update package X" / "bump uv" / "upgrade torch"** → `/veomni-uv-update`
