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
│   ├── veomni-new-op/
│   ├── veomni-uv-update/
│   └── create-pr/
├── knowledge/           # Shared knowledge base
│   ├── architecture.md
│   ├── constraints.md
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
- **uv.md** — dependency management architecture
