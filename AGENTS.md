# VeOmni Development Guide

**VeOmni** is a modular distributed training framework for multi-modality models (text, vision, audio, diffusion, omni) across GPUs, Ascend NPUs, and Cambricon MLUs.
- Python `>=3.11, <3.13` | `transformers==5.16.1` | FSDP2 only | Package: `veomni`
- **Language**: Match user's language in chat; all code comments and docstrings must be in English.

## Context Loading

Read on session start:
- `.agents/knowledge/constraints.md` — cross-cutting hard constraints (and index to deep-dive topics)
- `.agents/knowledge/architecture.md` — module map, trainer/runtime hierarchy, core flows, test/CI map

## VeOmni Rules

- **Cross-Modality Impact**: Changes in `BaseTrainer`, `data_collator`, `distributed/`, `models/registry.py`, or `configs/` keys affect all modalities — check callers, composed trainers (`TextDPOTrainer`, `DiTTrainer`), and `configs/` YAMLs before editing.
- **Standalone HF Modeling**: HF-derived models live in a single `veomni/models/transformers/<m>/modeling_<m>.py` with `# Upstream-Module:` and `# Base-Transformers-Version:` headers.
- **Minimal & Proportional**: Keep changes minimal and aligned with VeOmni's synchronous, single-threaded-per-rank runtime contract (no speculative defensive scaffolding or synthetic edge-case tests; see `/veomni-review`). One logical change per commit.
- **Keep Docs Lean**: Only cross-cutting invariants (2–4 lines each) belong in `constraints.md` / `architecture.md`. Put internal single-PR or single-model details in code docstrings or PR descriptions (or `.agents/knowledge/topics/` only for recurring traps) — do not add per-PR internal design essays or migration logs to `docs/`.
- **Stop When Stuck**: After 3+ failed approaches, stop patching symptoms, summarize findings, and ask the user.

## Setup & Commands

```bash
uv sync --extra gpu --dev          # hardware extras (mutually exclusive): gpu | npu | npu_aarch64; optional: --extra magi (SM100+), --extra float8, --extra hyper (NPU)
source .venv/bin/activate          # CPU-only hosts also use `--extra gpu --dev` (omit `--extra magi`)
make style                         # ruff fix + format
make quality                       # ruff check + format check (CI gate)
make check-agent-docs              # verify repo paths and /skill-name refs in agent docs
python3 scripts/ci/docs/check_doc_task_paths.py  # verify task paths in docs/
pytest tests/<mod>/                # focused tests
```

## PR & Commit Flow

1. Verify the change; update `docs/` or `configs/` only when user-facing usage, config fields, or public workflows change (no per-PR internal design docs).
2. Run `make quality` before every commit.
3. **Commit messages** describe the change, not the tool/agent that produced it (no `Co-Authored-By` trailers).
4. **PR title**: `[{modules}] {type}: {description}` (prepend `[BREAKING]` if breaking; allowed values in `.github/workflows/check_pr_title.yml`).
5. **Pre-PR review gate**: Run `/veomni-review` once over the branch diff before opening a PR (and before substantive updates — not per commit). `safe` / `needs-attention` → address and proceed; `risky` → report and wait.

## Skills

Skills live in `.agents/skills/<name>/SKILL.md` (auto-discovered from `description` or invoked via `/skill-name`):

| Task | Skill |
|---|---|
| Pre-PR code review gate | `/veomni-review` |
| Add new model, update `modeling_<m>.py`, or migrate GPU model to NPU | `/veomni-new-model` |
| Add / modify SeedOmni module or graph | `/veomni-seed-omni` |
| Add / optimize kernel in `veomni/ops/` | `/veomni-new-op` |
| Update `uv` / `torch` / `transformers` / lockfile | `/veomni-uv-update` |
| Create or update a pull request | `/create-pr` |
