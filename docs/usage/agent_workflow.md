# Agent Workflow Guide

VeOmni provides a skill-based workflow system that helps AI coding agents work on the project effectively. Skills follow the [Agent Skills](https://agentskills.io) open standard and work with any compatible agent (Cursor, Claude Code, Codex, Junie, Goose, etc.).

## Overview

The workflow consists of three layers:

```
AGENTS.md                      <- Entry point: principles, skill dispatch, commit flow
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
2. For each task, the agent selects the appropriate skill from the dispatch table (or auto-discovers it via the `description` frontmatter).
3. The agent reads the skill's `SKILL.md` and follows its step-by-step instructions.
4. Before committing, the agent runs the `/veomni-review` skill (a subagent code review).

**You don't need to do anything special** — just describe your task in natural language. You can also invoke a specific skill with `/skill-name` in chat (e.g., `/veomni-debug`).

### Examples

| What you say | Agent uses |
|---|---|
| "Add support for Llama 4" | `/veomni-new-model` |
| "Fix the OOM error in VLM training" | `/veomni-bugfix` or `/veomni-debug` |
| "Add a fused RoPE kernel" | `/veomni-new-op` |
| "Refactor the data collator" | `/veomni-refactor` |
| "Update torch to 2.10" | `/veomni-uv-update` |

## Directory Structure

### `.agents/skills/`

Each skill is a folder containing a `SKILL.md` file with YAML frontmatter (`name` and `description`):

```
.agents/skills/
├── veomni-feature/SKILL.md      # New feature development
├── veomni-bugfix/SKILL.md       # Clear bug with obvious fix
├── veomni-refactor/SKILL.md     # Restructuring without behavior change
├── veomni-debug/SKILL.md        # Complex debugging (unclear root cause)
├── veomni-review/SKILL.md       # Pre-commit code review (mandatory)
├── veomni-verify/SKILL.md       # Validate conclusions after investigation
├── veomni-new-model/SKILL.md    # Add a new model to VeOmni
├── veomni-new-op/SKILL.md       # Add a new kernel/operator
├── veomni-run-test/SKILL.md     # Execute and record test results
├── veomni-uv-update/SKILL.md    # Dependency management with uv
└── veomni-housekeeping/SKILL.md # Recover state after context loss
```

The `description` field in frontmatter tells the agent when to apply the skill. Agents that support auto-discovery (Cursor, Claude Code) will offer the relevant skill automatically based on the task description.

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
- `skills-reusable-only.mdc` — enforce Agent Skills standard format in `.agents/skills/`

## Subagent Skills

`/veomni-review` and `/veomni-verify` launch a **separate AI subagent** for independent evaluation:
- **Review**: receives only the diff + constraints (not the developer's reasoning) to avoid confirmation bias
- **Verify**: acts as an adversary trying to disprove a conclusion

## Adding a New Skill

1. Create `.agents/skills/<skill-name>/SKILL.md` with YAML frontmatter:

```yaml
---
name: skill-name
description: "When to use this skill. Trigger words and scenarios."
---
```

2. Add the skill to the dispatch table in `AGENTS.md`.
3. Add it to the Skill Index in `.agents/skills/README.md`.
4. If the skill needs domain knowledge, add it to `.agents/knowledge/`.
5. Optional: add `scripts/`, `references/`, or `assets/` subdirectories.

See the [Agent Skills specification](https://agentskills.io/specification) for the full format.

## Adding Domain Knowledge

1. Create or edit a `.md` file in `.agents/knowledge/`.
2. Reference it from the Context Loading section in `AGENTS.md`.
3. If the knowledge contains hard rules, add them to `constraints.md`.

## Commit Flow

The agent workflow enforces a structured commit flow:

```
Code Change -> /veomni-review (subagent) -> Verdict
                                             |
                                     safe -> commit
                              needs-attention -> fix, then commit
                                     risky -> report to user, wait
```

Additional gates:
- `make quality` must pass (ruff check + format)
- Commit messages must not mention AI/Claude
- PR title must follow `[{modules}] {type}: {description}` format
