---
name: veomni-review
description: "Pre-commit code review gate. Required before committing any change to Python under veomni/, tasks/ or tests/, and any change to CI workflows, pyproject.toml or configs/. Also trigger proactively when a change spans multiple files, touches shared infrastructure (BaseTrainer, distributed, model loading, data pipeline, ops dispatch), or you are unsure a fix is safe. The review launches a subagent that checks implementation quality, multi-file consistency, and known constraint violations, then rates the change as safe/needs-attention/risky."
---

## When this gate applies

| Change | Review |
|--------|--------|
| Python under `veomni/`, `tasks/`, `tests/` | Required |
| `.github/workflows/`, `pyproject.toml`, `uv.lock`, `docker/`, `configs/` | Required |
| Docs, comments, or `.agents/` knowledge and skills only | Skip — self-check instead: verify every repo path, config key and version you assert actually exists |
| A revert, or re-applying a diff a reviewer already approved | Skip |

Skipping means skipping the subagent, not skipping verification. Say which
branch you took, so the reader knows a review happened or why it didn't.

## Steps

1. Run `git diff` (staged + unstaged) to capture the full diff.
2. Read `.agents/knowledge/constraints.md` for known constraints.
3. **Launch a review subagent** with your agent's subagent/task mechanism (see
   the prompt below). The subagent receives only the diff + constraints — NOT
   your reasoning — to avoid confirmation bias. If the change is already
   committed on a feature branch, point it at `git diff main...HEAD` instead of
   pasting the diff.
4. Act on the verdict.

| Verdict | Action |
|---------|--------|
| **safe** | Proceed to commit |
| **needs-attention** | Address listed issues, then commit |
| **risky** | Output the report, do NOT commit, wait for user |

5. Run `make quality` before the final commit.

## Subagent Launch

Launch a subagent with this prompt. Use whatever the running agent calls it —
`Task`, `spawn_agent`, or an equivalent — and give it read-only access to the
repo so it can verify claims against the actual files.

```
You are a code reviewer for VeOmni, a distributed multi-modality training framework. Your job is to find problems in the following diff. You are NOT validating the author's intent — you are looking for bugs, risks, and constraint violations.

## Diff
<paste full git diff here>

## Known Constraints
<paste constraints.md content here>

## Review Checklist

For each changed file, check:

### Implementation Quality
- Hidden risks or edge cases not handled?
- Simpler alternative that achieves the same result?
- Boundary conditions (tensor shapes, distributed rank handling, gradient accumulation steps)?
- Does the fix depend on downstream code to "clean up"?

### Multi-file Consistency
- If a Trainer method changed, do all subclasses need matching changes?
- If model loading changed, are configs and parallel plans updated?
- If data collator changed, do all modalities still work?
- If distributed code changed, are the FSDP2, sequence-parallel and ExtraParallel/EP paths all handled? (FSDP1 no longer exists — a diff that adds an FSDP1 branch is itself a finding.)
- If a trainer lifecycle hook changed, do the composed trainers that override `forward_backward_step()` (`TextDPOTrainer`, `DiTTrainer`) still get it?

### Constraint Violations
- Does this violate any entry in the known-constraints list?
- Does this repeat a pattern that previously caused bugs?

### VeOmni-Specific Checks
- PR title format: `[{modules}] {type}: {description}`?
- All comments and docstrings in English?
- No auto-generated files (`veomni/models/transformers/*/generated/`) edited directly?
- Tests: does the diff extend an existing CI-enumerated test, or add a new file that is actually wired into the unit-test workflows? A new test file outside `tests/data/` and `tests/ops/` that no workflow lists will never run. Conversely, a workflow line added for a file under `tests/data/` or `tests/ops/` is redundant. See `.agents/knowledge/testing.md`.
- Ruff-compliant (`make quality` passes)?

## Output

### Verdict: safe / needs-attention / risky

### Findings (for needs-attention or risky)
For each issue:
- **File**: path:line
- **Concern**: what could go wrong
- **Suggestion**: what to do instead
```

## After Commit

- Run `make quality` to confirm ruff compliance.
- Verify PR title follows `[{modules}] {type}: {description}` format.
