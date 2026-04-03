# VeOmni Agent Skills

Reusable workflow definitions for AI coding agents working on VeOmni. Skills follow the [Agent Skills](https://agentskills.io) open standard and are auto-discovered by compatible agents (Cursor, Claude Code, Codex, Junie, etc.).

## Structure

Each skill is a folder containing a `SKILL.md` with YAML frontmatter:

```
.agents/skills/
├── veomni-develop/
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
4. Optional: add `scripts/`, `references/`, or `assets/` subdirectories.

See the [Agent Skills specification](https://agentskills.io/specification) for the full format.

## Skill Index

| Skill | Description |
|-------|-------------|
| `veomni-develop` | Feature development and refactoring — VeOmni-specific impact analysis and safety checklist |
| `veomni-debug` | Bug fix and debugging — quick path for simple fixes, full protocol for complex issues |
| `veomni-review` | Pre-commit code review via subagent (mandatory gate) |
| `veomni-new-model` | Adding a new model to VeOmni (patchgen, parallel plan, registry) |
| `veomni-new-op` | Adding a new optimized kernel/operator to veomni/ops/ |
| `veomni-uv-update` | Dependency management with uv (version bumps, torch, lockfile) |
| `create-pr` | Create a pull request — handles uncommitted changes, generates CI-compliant title and description |
