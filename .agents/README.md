# .agents/ — Shared Agent Configuration

Reusable skills and knowledge for AI coding agents working on VeOmni. Follows the [Agent Skills](https://agentskills.io) open standard, so any agent that implements it can use these as-is.

## Structure

```
.agents/
├── skills/              # Workflow definitions (each skill = folder with SKILL.md)
│   ├── veomni-develop/
│   ├── veomni-debug/
│   ├── veomni-review/
│   ├── veomni-new-model/
│   ├── veomni-patchgen-model/
│   ├── veomni-new-op/
│   ├── veomni-uv-update/
│   ├── veomni-profile/
│   └── create-pr/
├── knowledge/           # Shared knowledge base
│   ├── architecture.md
│   ├── constraints.md
│   ├── cpu_only_env.md
│   ├── multimodal_metadata.md
│   ├── testing.md
│   └── uv.md
├── setup_agent.sh       # Bootstrap script (see below)
└── README.md
```

## Quick Start

Some agents read the `.agents/` directory natively and will auto-discover skills and knowledge — no extra setup needed.

For an agent that insists on its own dotfile directory, run the bootstrap script with that agent's name:

```bash
bash .agents/setup_agent.sh <agent-name>
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
- **cpu_only_env.md** — what can be verified without a GPU/NPU
- **multimodal_metadata.md** — canonical multimodal metadata keys and ownership
- **testing.md** — how CI selects tests; whether a change needs one and where it goes
- **uv.md** — dependency management architecture

## Keeping these docs honest

`make check-agent-docs` (run in CI by the **Check doc task paths** workflow)
resolves, in `AGENTS.md`, `.agents/**/*.md` and `.cursor/rules/*`: every
repo-root-relative path, every bare `*.yml` workflow name, and every skill
reference. Skill references are checked across the whole tracked tree, since a
renamed skill also leaves dangling mentions in code comments. Use
`<angle brackets>` for placeholders so they are skipped, and note that a bare
`.py` name is deliberately not resolved — docs name upstream transformers
modules and illustrative filenames that do not exist here. It cannot check
prose claims about behaviour: when you edit a skill, re-read the file it
describes.
