# .agents/ — Shared Agent Configuration

Reusable skills and layered knowledge for AI coding agents working on VeOmni, following the [Agent Skills](https://agentskills.io) open standard.

## Structure

```text
.agents/
├── skills/                     # Focused workflows (folder + SKILL.md)
│   ├── veomni-review/          # Pre-PR code review gate
│   ├── veomni-new-model/       # Model integration, modeling_<m>.py, weight conversion, GPU→NPU
│   ├── veomni-seed-omni/       # SeedOmni modules, graphs, per-module parallelism
│   ├── veomni-new-op/          # Tensor-level ops/kernels under veomni/ops/
│   ├── veomni-uv-update/       # uv / torch / transformers / lockfile updates
│   └── create-pr/              # Pull request drafting and submission
├── knowledge/                  # Layered domain knowledge
│   ├── constraints.md          # Session start: cross-cutting hard constraints
│   ├── architecture.md         # Session start: module map, hierarchy, core flows
│   ├── testing.md              # On demand: CI selection, launch plans, test decision rule
│   ├── multimodal_metadata.md  # On demand: VLM/Omni metadata precompute contract
│   ├── veomni-seed-omni.md     # On demand: SeedOmni architecture & constraints
│   └── topics/                 # On demand: subsystem traps not covered in docs/design/
│       ├── qat_and_ops.md
│       ├── distributed.md
│       └── flex_attention.md
├── setup_agent.sh              # Symlink bootstrap for dotfile-specific agents
└── README.md
```

## Quick Start

Agents supporting `.agents/` auto-discover it natively. For agents expecting `.<agent_name>/`:

```bash
bash .agents/setup_agent.sh <agent-name>
```

Creates `.<agent_name>/`, symlinks `skills/`, `knowledge/`, and `README.md`, and adds it to `.git/info/exclude`.

## Authoring & Hygiene Rules

- **Keep startup docs lean**: `AGENTS.md`, `constraints.md`, and `architecture.md` hold only cross-cutting rules and pointers. Keep internal single-PR details in code docstrings or PR descriptions (and `.agents/knowledge/topics/` only for recurring traps), rather than adding per-PR docs to `docs/`.
- **Keep skills concise**: each skill is `.agents/skills/<skill-name>/SKILL.md` with YAML frontmatter (`name` matching folder, `description` for auto-discovery). Aim for ~50–100 lines; move conditional topics into `references/` and link to `constraints.md` / `docs/design/` rather than duplicating them. Update the dispatch table in [`AGENTS.md`](../AGENTS.md) when adding or renaming a skill.
- **Automated path & skill gate**: run `make check-agent-docs` (`scripts/ci/docs/check_agent_doc_paths.py`, enforced in CI) after editing agent docs. Use `<angle brackets>` for placeholders.
