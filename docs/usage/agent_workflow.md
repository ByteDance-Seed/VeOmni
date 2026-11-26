# Agent Workflow Guide

VeOmni ships a lightweight, skill-based workflow for AI coding agents following the [Agent Skills](https://agentskills.io) open standard (no VeOmni-specific plugin required).

## How It Works

- **Entry point**: `AGENTS.md` (with `CLAUDE.md` symlinked to it) defines core rules, commands, PR/commit gates, and the skill dispatch table.
- **Layered knowledge (`.agents/knowledge/`)**: agents load only `constraints.md` and `architecture.md` on session start (~20 KB total); specialized docs (`testing.md`, `multimodal_metadata.md`, `veomni-seed-omni.md`, `topics/*.md`) are loaded on demand.
- **Focused skills (`.agents/skills/`)**: auto-discovered from YAML frontmatter `description` or invoked manually via `/skill-name` (see `.agents/README.md` for the full tree and authoring rules).
- **IDE rules (`.cursor/rules/`)**: `no-section-divider-comments.mdc` and `skills-reusable-only.mdc`.

## Common Tasks

| What you say | Skill |
|---|---|
| "Add support for Llama 4" / "Update `modeling_qwen3.py`" / "Migrate Gemma 3 to Ascend NPU" | `/veomni-new-model` |
| "Add a SeedOmni module or update `training_graph`" | `/veomni-seed-omni` |
| "Add a fused RoPE kernel" | `/veomni-new-op` |
| "Update torch, transformers, or uv lockfile" | `/veomni-uv-update` |
| "Review my branch before opening a PR" | `/veomni-review` |
| "Submit the current branch as a PR" | `/create-pr` |

## Commit & Review Flow

```text
each commit    -> make quality + focused verification
before opening or substantively updating a PR
               -> /veomni-review -> safe / needs-attention (fix first) / risky (report & wait)
```

Additional gates:
- `make quality` (`ruff check` + `ruff format --check`) on every commit.
- `make check-agent-docs` after editing `AGENTS.md` or `.agents/`.
- Commit messages describe the change, not the assistant/agent (no `Co-Authored-By` trailers).
- PR titles match `[{modules}] {type}: {description}` (`.github/workflows/check_pr_title.yml`).
