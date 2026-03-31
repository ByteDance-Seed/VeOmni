# VeOmni Agent Skills

Reusable workflow definitions for AI coding agents working on VeOmni. Skills follow the [Agent Skills](https://agentskills.io) open standard and are auto-discovered by compatible agents (Cursor, Claude Code, Codex, Junie, etc.).

## Structure

Each skill is a folder containing a `SKILL.md` with YAML frontmatter:

```
.agents/skills/
├── veomni-feature/
│   └── SKILL.md          # name + description frontmatter, then instructions
├── veomni-debug/
│   └── SKILL.md
└── ...
```

Agents use the `description` field to decide when a skill is relevant. Users can also invoke skills manually with `/skill-name` in chat.

## Adding a Skill

1. Create `.agents/skills/<skill-name>/SKILL.md` with `name` and `description` frontmatter.
   - `name` must match the folder name (lowercase, hyphens only).
   - `description` should explain what the skill does and when to use it.
2. Add the skill to the dispatch table in `AGENTS.md`.
3. If the skill requires domain knowledge, add it to `.agents/knowledge/`.
4. Optional: add `scripts/`, `references/`, or `assets/` subdirectories for supporting files.

See the [Agent Skills specification](https://agentskills.io/specification) for the full format.

## Skill Index

| Skill | Description |
|-------|-------------|
| `veomni-feature` | New feature development workflow |
| `veomni-bugfix` | Lightweight bugfix for straightforward bugs |
| `veomni-refactor` | Safe refactoring with baseline verification |
| `veomni-debug` | Four-phase debugging protocol for complex issues |
| `veomni-review` | Pre-commit code review via subagent |
| `veomni-verify` | Adversarial verification of conclusions |
| `veomni-new-model` | Adding a new model to VeOmni |
| `veomni-new-op` | Adding a new optimized kernel/operator to veomni/ops/ |
| `veomni-run-test` | Orchestrated test execution and result recording |
| `veomni-uv-update` | Dependency management with uv (version bumps, torch, lockfile) |
| `veomni-housekeeping` | Post-compaction state verification |
