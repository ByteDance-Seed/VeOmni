# .agents/ — Shared Agent Configuration

Reusable skills and knowledge for AI coding agents working on VeOmni. Follows the [Agent Skills](https://agentskills.io) open standard — compatible with Gemini, Codex, Claude Code, Cursor, and other agents.

## Structure

```
.agents/
├── skills/              # Workflow definitions (each skill = folder with SKILL.md)
│   ├── veomni-develop/
│   ├── veomni-debug/
│   ├── veomni-review/
│   ├── veomni-new-model/
│   ├── veomni-migrate-transformers-v5/
│   ├── veomni-new-op/
│   ├── veomni-uv-update/
│   ├── veomni-profile/
│   └── create-pr/
├── knowledge/           # Shared knowledge base
│   ├── architecture.md
│   ├── constraints.md
│   ├── multimodal_metadata.md
│   └── uv.md
├── setup_agent.sh       # Bootstrap script (see below)
└── README.md
```

## Quick Start

Some agents natively support the `.agents/` directory and will auto-discover skills and knowledge — no extra setup needed.

For agents that require their own dotfile directory, run the bootstrap script:

```bash
# Example: set up for Gemini (replace with your agent name)
bash .agents/setup_agent.sh gemini
```

This will:

1. Create a `.<agent_name>/` directory in the project root
2. Symlink `skills/`, `knowledge/`, and `README.md` from `.agents/` into it
3. Add `.<agent_name>/` to `.git/info/exclude` (local-only, not committed)

## Skills

See [skills/README.md](skills/README.md) for the full skill index and how to add new ones.

## Knowledge

The `knowledge/` directory contains domain-specific context loaded by agents on session start:

- **architecture.md** — module map, trainer hierarchy, data flow
- **constraints.md** — hard constraints checked before any code change
- **multimodal_metadata.md** — canonical multimodal metadata keys and ownership
- **testing.md** — how CI selects tests; whether a change needs one and where it goes
- **uv.md** — dependency management architecture

## Keeping these docs honest

`make check-agent-docs` (run in CI by the **Check doc task paths** workflow)
verifies that every repo path and skill name referenced from `AGENTS.md`,
`.agents/**/*.md` and `.cursor/rules/*` actually exists. Use `<angle brackets>`
for placeholders so they are skipped. It cannot check prose claims about
behaviour — when you edit a skill, re-read the file it describes.
