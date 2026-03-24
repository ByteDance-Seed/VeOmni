# VeOmni Agent Skills

Reusable workflow definitions for AI coding agents working on VeOmni.

## How It Works

```
.agents/skills/              → Skill definitions (read directly by agents)
.agents/knowledge/           → Domain knowledge (constraints, architecture)
```

Skills are referenced from `CLAUDE.md`. Agents read the relevant skill file when a matching task is detected.

## Skill Types

- **`<skill-name>.md`** — Lightweight workflow (single file, no frontmatter)
- **`<skill-name>/SKILL.md`** — Complex workflow with YAML frontmatter (`name`, `description`)

## Adding a Skill

1. Create `<skill-name>/SKILL.md` or `<skill-name>.md` in this directory
2. Add the skill to the dispatch table in `CLAUDE.md`
3. If the skill requires domain knowledge, add it to `.agents/knowledge/`

## Skill Index

| Skill | Type | Description |
|-------|------|-------------|
| `veomni-feature` | `.md` | New feature development workflow |
| `veomni-bugfix` | `.md` | Lightweight bugfix for straightforward bugs |
| `veomni-refactor` | `.md` | Safe refactoring with baseline verification |
| `veomni-debug` | `SKILL.md` | Four-phase debugging protocol for complex issues |
| `veomni-review` | `SKILL.md` | Pre-commit code review via subagent |
| `veomni-verify` | `SKILL.md` | Adversarial verification of conclusions |
| `veomni-new-model` | `SKILL.md` | Adding a new model to VeOmni |
| `veomni-new-op` | `SKILL.md` | Adding a new optimized kernel/operator to veomni/ops/ |
| `veomni-run-test` | `SKILL.md` | Orchestrated test execution and result recording |
| `veomni-uv-update` | `SKILL.md` | Dependency management with uv (version bumps, torch, lockfile) |
| `veomni-housekeeping` | `SKILL.md` | Post-compaction state verification |
