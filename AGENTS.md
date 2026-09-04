# VeOmni Development Guide

> Instructions for AI coding agents working on this repository.

**VeOmni** is a modular distributed training framework for multi-modality models (text, vision, audio, diffusion, omni) across various accelerators (GPUs, NPUs). Developed by ByteDance Seed Team.

- Homepage: https://github.com/ByteDance-Seed/VeOmni
- Python: `>=3.11, <3.13`
- Package: `veomni`

**Language**: Match user's language (English).

## Context Loading

On session start, read the following:
- `.agents/knowledge/constraints.md` — hard constraints to check before any code change
- `.agents/knowledge/architecture.md` — module map, trainer hierarchy, data flow
- `.agents/knowledge/uv.md` — dependency management architecture (uv, extras, lockfile)
- `.agents/knowledge/cpu_only_env.md` — what can be verified without a GPU/NPU; read when running on a CPU-only machine

Read on demand:
- `.agents/knowledge/testing.md` — how CI selects tests, and how to decide whether a change needs one; read before adding or wiring a test
- `.agents/knowledge/multimodal_metadata.md` — multimodal metadata precompute contract; read before touching VLM/Omni collators, ViT forwards, or the model metadata hooks

---

## Core Principles

- **Challenge First, Execute Second**: Spot logic flaws or simpler alternatives? Raise concerns before executing.
- **Explain, Don't Assume**: Explain **why** (motivation, tradeoffs), not just what. Cite files and line numbers.
- **Ask When Stuck**: 3+ approaches fail? Stop, summarize, ask user. No hacks.
- **Search Before You Act**: On unexpected behavior, search codebase + check constraints + review `git log` before attempting fixes.
- **Planning Discipline**: Complex tasks (multi-file, >30 min) -> write a plan with the agent's todo/plan tool. The plan must state which skills will be used (e.g. `/veomni-develop` + `/veomni-review`). Simple tasks -> just do them.
- **Cross-modality Awareness**: Changes in shared code (`BaseTrainer`, `data_collator`, `distributed/`) affect all modalities.
- **No Patchgen Edits**: Never edit files under `veomni/models/transformers/*/generated/`.

---

## Setup

```bash
uv sync --extra gpu --dev
source .venv/bin/activate
```

This installs `transformers==5.9.0` via the `transformers-stable` dependency
group. `gpu` / `npu` / `npu_aarch64` are the only extras — each a complete
superset, mutually exclusive. New code must target transformers v5 and FSDP2.
See `.agents/knowledge/uv.md` and `.agents/knowledge/constraints.md`.

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
- GitHub PRs are also reviewed by CodeRabbit (`.coderabbit.yaml`). On an existing PR, comment `@coderabbitai review` or `@coderabbitai full review`.

---

## Commit Flow

1. Complete and verify the change.
2. Update related documentation: `docs/`, `README.md`, `.agents/knowledge/`, config examples — if the change introduces, modifies, or removes any API, config field, or workflow.
3. Run `/veomni-review` skill (subagent code review).
4. **safe** -> commit. **risky** -> report to user, wait for approval.
5. Each fix -> immediate commit. Do not batch unrelated changes.
6. Run `make quality` before every commit.
7. **Commit messages describe the change, not the tool that produced it** — do not name the assistant or agent that wrote it, and no `Co-Authored-By` trailers. (Naming a *model being added* is expected — that is the change.)
8. **Skill gap check**: If the task didn't match any existing skill, briefly assess after completion: Was this a one-off, or a repeatable pattern? If repeatable, suggest creating a new skill to the user.

---

## Skills

Skills follow the [Agent Skills](https://agentskills.io) open standard. Each skill is a folder in `.agents/skills/<name>/` containing a `SKILL.md` with YAML frontmatter (`name`, `description`). Agents that implement the standard auto-discover them from the `description`; they can also be invoked manually with `/skill-name` in chat.

| Task | Skill |
|------|-------|
| Feature / refactoring | `/veomni-develop` |
| Bug fix / debugging | `/veomni-debug` |
| Code review (pre-commit) | `/veomni-review` |
| Add new model | `/veomni-new-model` |
| Write or refresh a model's patchgen modeling | `/veomni-patchgen-model` |
| Add new op/kernel | `/veomni-new-op` |
| Update dependencies (uv) | `/veomni-uv-update` |
| Performance profiling | `/veomni-profile` |
| Create or update a pull request | `/create-pr` |

### Quick Decision Guide

- **"Add support for model X"** → `/veomni-new-model`
- **"Write a patch_gen_config" / "regenerate the generated modeling" / "add NPU patchgen" / "port X to patchgen"** → `/veomni-patchgen-model` (also the modeling step inside `/veomni-new-model`)
- **"Add a new kernel / fused op"** → `/veomni-new-op`
- **"Fix this error" / "training hangs" / "wrong results"** → `/veomni-debug`
- **"Add a new capability" / "refactor" / "clean up"** → `/veomni-develop`
- **"Update package X" / "bump uv" / "upgrade torch"** → `/veomni-uv-update`
- **"Analyze this trace" / "why is training slow" / "profile" / "MFU"** → `/veomni-profile`
- **"Create a PR" / "submit PR"** → `/create-pr`
