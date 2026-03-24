# Agent Workflow Guide

VeOmni provides a skill-based workflow system that helps AI coding agents (Cursor, Claude Code, etc.) work on the project effectively. This document explains how to use it.

## Overview

The workflow consists of three layers:

```
AGENTS.md / CLAUDE.md          <- Entry point: principles, skill dispatch, commit flow
.agents/skills/                <- Skills: step-by-step workflows for common tasks
.agents/knowledge/             <- Knowledge: constraints, architecture, dependency info
.cursor/rules/                 <- IDE rules: Cursor-specific coding conventions
```

When an agent opens the project, it reads `AGENTS.md` (or its symlink `CLAUDE.md`) to understand:
- **What constraints to follow** before making any change
- **Which skill to use** for the task at hand
- **How to commit** (mandatory code review gate)

## Quick Start

### For AI Agent Users

If you are using Cursor or another AI coding tool on this project, the workflow activates automatically:

1. The agent reads `AGENTS.md` on session start.
2. For each task, the agent selects the appropriate skill from the dispatch table.
3. The agent reads the skill file and follows its step-by-step instructions.
4. Before committing, the agent runs the `veomni-review` skill (a subagent code review).

**You don't need to do anything special** — just describe your task in natural language.

### Examples

| What you say | Agent uses |
|---|---|
| "Add support for Llama 4" | `veomni-new-model` skill |
| "Fix the OOM error in VLM training" | `veomni-bugfix` or `veomni-debug` skill |
| "Add a fused RoPE kernel" | `veomni-new-op` skill |
| "Refactor the data collator" | `veomni-refactor` skill |
| "Update torch to 2.10" | `veomni-uv-update` skill |

## Directory Structure

### `.agents/skills/`

Each skill is a markdown file defining a step-by-step workflow:

| Skill | File | When to Use |
|-------|------|-------------|
| `veomni-feature` | `veomni-feature.md` | New feature development |
| `veomni-bugfix` | `veomni-bugfix.md` | Clear bug with obvious fix |
| `veomni-refactor` | `veomni-refactor.md` | Restructuring without behavior change |
| `veomni-debug` | `veomni-debug/SKILL.md` | Complex debugging (unclear root cause) |
| `veomni-review` | `veomni-review/SKILL.md` | Pre-commit code review (mandatory) |
| `veomni-verify` | `veomni-verify/SKILL.md` | Validate conclusions after investigation |
| `veomni-new-model` | `veomni-new-model/SKILL.md` | Add a new model to VeOmni |
| `veomni-new-op` | `veomni-new-op/SKILL.md` | Add a new kernel/operator |
| `veomni-run-test` | `veomni-run-test/SKILL.md` | Execute and record test results |
| `veomni-uv-update` | `veomni-uv-update/SKILL.md` | Dependency management with uv |
| `veomni-housekeeping` | `veomni-housekeeping/SKILL.md` | Recover state after context loss |

**Skill types:**
- **`.md` files** — lightweight workflows (feature, bugfix, refactor)
- **`SKILL.md` in a directory** — complex workflows with YAML frontmatter containing trigger descriptions

### `.agents/knowledge/`

Domain knowledge that agents should read before making changes:

| File | Content |
|------|---------|
| `constraints.md` | 20 hard constraints — violating any one causes bugs or crashes |
| `architecture.md` | Module map, trainer hierarchy, data flow, model loading flow |
| `uv.md` | Dependency management architecture (uv, extras, lockfile, torch sources) |

### `.cursor/rules/`

Cursor IDE rules (auto-applied when editing matching files):

- `no-section-divider-comments.mdc` — no decorative `# ----` banners in Python
- `skills-reusable-only.mdc` — `.agents/skills/` only for reusable workflows

## How Skills Work

### Lightweight Skills (`.md`)

Simple markdown files with a step-by-step recipe. Example flow for `veomni-bugfix`:

```
1. Reproduce  ->  2. Identify Root Cause  ->  3. Fix  ->  4. Verify  ->  5. Commit
```

### Complex Skills (`SKILL.md`)

YAML frontmatter with `name` and `description` (trigger phrases), followed by a phased protocol. Example for `veomni-debug`:

```
Phase 1: Root Cause Investigation    (with TodoWrite tracking)
Phase 2: Pattern Analysis
Phase 3: Hypothesis and Testing
Phase 4: Implementation
Phase 5: Knowledge Capture           (update constraints if new rule found)
```

### Subagent Skills

`veomni-review` and `veomni-verify` launch a **separate AI subagent** for independent evaluation:
- **Review**: receives only the diff + constraints (not the developer's reasoning) to avoid confirmation bias
- **Verify**: acts as an adversary trying to disprove a conclusion

## Adding a New Skill

1. Create `<skill-name>.md` or `<skill-name>/SKILL.md` in `.agents/skills/`.
2. Add the skill to the dispatch table in `AGENTS.md`.
3. Add it to the Skill Index in `.agents/skills/README.md`.
4. If the skill needs domain knowledge, add it to `.agents/knowledge/`.

## Adding Domain Knowledge

1. Create or edit a `.md` file in `.agents/knowledge/`.
2. Reference it from the Context Loading section in `AGENTS.md`.
3. If the knowledge contains hard rules, add them to `constraints.md`.

## Commit Flow

The agent workflow enforces a structured commit flow:

```
Code Change -> veomni-review (subagent) -> Verdict
                                             |
                                     safe -> commit
                              needs-attention -> fix, then commit
                                     risky -> report to user, wait
```

Additional gates:
- `make quality` must pass (ruff check + format)
- Commit messages must not mention AI/Claude
- PR title must follow `[{modules}] {type}: {description}` format
