---
name: veomni-review
description: "Pre-PR code review gate. Run before opening a pull request, and again before pushing a substantive update to an open one — not per commit. Required when the PR's branch diff touches Python under veomni/, tasks/ or tests/, or CI workflows, pyproject.toml, uv.lock, docker/ or configs/. Docs, comments and .agents/-only changes use the self-check below. Launches a read-only subagent that checks quality, consistency, anti-bloat, and constraint violations, returning safe/needs-attention/risky."
---

## When This Gate Applies

**Once per PR (and before substantive updates to an open PR) — not per commit.** Commits are gated by `make quality` and local verification.

| Branch diff touches | Review |
|---|---|
| Python under `veomni/`, `tasks/`, `tests/` | Required (subagent) |
| `.github/workflows/`, `pyproject.toml`, `uv.lock`, `docker/`, `configs/` | Required (subagent) |
| Docs, comments, or `.agents/` only | Skip subagent — self-check repo paths, config keys, and versions |
| Clean, exact revert or reapplication of a previously approved diff (no extra edits or conflict resolutions) | Skip |

State explicitly which branch of the table applied.

## Steps

1. Capture the full branch diff against `<base>` (usually `main`, or parent branch for stacked PRs):
   ```bash
   git diff <base>...HEAD
   ```
   If uncommitted work is also part of the review, make untracked files visible without staging content: `git add -N . && git diff HEAD`.
2. Launch a read-only review subagent with the prompt below. Give it only the diff command and `.agents/knowledge/constraints.md` — **not** your own reasoning (avoids confirmation bias).
3. Act on the verdict:
   - **`safe`**: open or update the PR.
   - **`needs-attention`**: fix listed findings (including deleting any flagged defensive bloat), then proceed.
   - **`risky`**: report findings to the user and wait — do **not** open/push the PR.
4. Run `make quality` before pushing.

## Subagent Prompt

Give the subagent read-only repo access so it can verify claims against live files:

```text
You are a code reviewer for VeOmni, a distributed multi-modality training framework. Your job is to find bugs, risks, bloat, and constraint violations — not to validate the author's intent.

## Inputs
- Diff: run `git diff <base>...HEAD` (include `git diff HEAD` only when uncommitted changes are part of the requested review)
- Constraints: read `.agents/knowledge/constraints.md` (and any relevant file under `.agents/knowledge/topics/` or `.agents/knowledge/veomni-seed-omni.md` for touched subsystems)

## Review Checklist

### 1. Implementation Quality
- Real bugs or boundary issues (tensor shapes, distributed ranks, grad accumulation, dtypes) in real training/inference workflows?
- Simpler alternative achieving the same result with substantially less code?
- Does the change depend on downstream code to silently clean up?

### 2. Anti-Bloat & Proportionality (CRITICAL)
Flag speculative defensive scaffolding or synthetic unit tests for scenarios outside VeOmni's runtime contract as `needs-attention` and recommend **deleting** them (never promote them into `.agents/knowledge/`):
- Thread locks, `asyncio`/thread-ID ownership checks, or epoch tokens on synchronous single-threaded-per-rank trainer/model/callback/op state.
- Cross-rank SHA-256 digests, pickle checks, or preflight `all_gather`/`all_reduce` collectives defending against SP/DP ranks disagreeing on config or shared batch metadata.
- Tensor-cloning guards checking whether trusted internal hooks (`metadata_collate_func`, callbacks) mutated inputs, or wrappers guarding against user keys colliding with private `_veomni_*` keys.
- Provenance state machines or DP solvers for 0-token samples, mid-sample `position_ids` resets, or multi-row packed batches (`PackingCollator` packs 1 row).
- Handling/testing `complex64`, `.conj()`, rank-asymmetric `torch._neg_view`, deliberate FP16 overflow to `NaN`, custom `sys.meta_path` finders, or `isinstance(x, bool)` on typed internal helpers.
- Meta-tests parsing AST/`inspect.getsource` of trainer/test helpers or testing another test's `pytest.skip`.
- Disproportionate diff size (e.g. 40-line fix wrapped in 500+ lines of defensive scaffolding).

### 3. Multi-File Consistency & Constraints
- Trainer lifecycle changes: do composed overrides (`TextDPOTrainer`, `DiTTrainer`) and RL subclasses stay consistent?
- Model loading / data collator / distributed changes: are FSDP2, Ulysses SP, ExtraParallel/EP, DDP, and all modalities handled?
- Does the diff violate any entry in `.agents/knowledge/constraints.md` or relevant topic constraints?

### 4. VeOmni Gates & Documentation
- PR title matches `[{modules}] {type}: {description}` (`.github/workflows/check_pr_title.yml`); all comments/docstrings in English; `make quality` passes.
- HF-derived `veomni/models/transformers/*/modeling_*.py` preserve `# Upstream-Module:` / `# Base-Transformers-Version:` headers, instance-local `VeomniOp`, and multi-backend safety.
- Tests: extends an existing CI-enumerated test, sits in a wholesale dir (`tests/{ci,arguments,data,checkpoints,ops,models,seed_omni}/`, plus `tests/parallel/context_parallel/` on GPU), or is added to `scripts/ci/accelerator/suites/{gpu,npu}.json` (`.agents/knowledge/testing.md`).
- Docs: update user-facing `docs/` only when user-facing usage, CLI/YAML config, or public APIs change (if a new page is added under `docs/`, wire it into a `docs/index.md` toctree so `make -C docs html SPHINXOPTS=-W` passes). Do **not** demand per-PR internal design essays, migration logs, or brittle `#L<line>` anchors — keep internal rationale in code docstrings and PR descriptions. When docs change, run `python3 scripts/ci/docs/check_doc_task_paths.py` and `make check-agent-docs`.

## Output
### Verdict: safe / needs-attention / risky
### Findings (for needs-attention or risky)
For each issue:
- **File**: path:line
- **Concern**: what could go wrong (or why it is unnecessary bloat)
- **Suggestion**: concrete fix or deletion
```
