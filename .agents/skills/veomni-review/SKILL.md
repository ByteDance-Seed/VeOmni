---
name: veomni-review
description: "Pre-PR code review gate. Run once before opening a pull request, and again before pushing a substantive update to an open one — not per commit. Required when the PR's branch diff touches Python under veomni/, tasks/ or tests/, or CI workflows, pyproject.toml, uv.lock, docker/ or configs/. Also trigger proactively when a change spans multiple files, touches shared infrastructure (BaseTrainer, distributed, model loading, data pipeline, ops dispatch), or you are unsure a fix is safe. The review launches a subagent that checks implementation quality, multi-file consistency, and known constraint violations, then rates the change as safe/needs-attention/risky."
---

## When this gate applies

**Once per PR, not once per commit.** The unit that lands is the pull request,
so that is the unit worth reviewing: the reviewer sees the whole change instead
of a slice of it, and a branch of ten commits costs one review rather than ten.
Run it again before pushing a substantive update to an open PR — not for a
typo fix or a rebase.

Commits stay cheap: `make quality` and your own verification still gate every
commit, and nothing stops you invoking this mid-branch when a change worries
you. It is the *obligation* that moved, not the option.

| The PR's branch diff touches | Review |
|--------|--------|
| Python under `veomni/`, `tasks/`, `tests/` | Required |
| `.github/workflows/`, `pyproject.toml`, `uv.lock`, `docker/`, `configs/` | Required |
| Docs, comments, or `.agents/` knowledge and skills only | Skip — self-check instead: verify every repo path, config key and version you assert actually exists |
| A revert, or re-applying a diff a reviewer already approved | Skip |

Skipping means skipping the subagent, not skipping verification. Say which
branch you took, so the reader knows a review happened or why it didn't.

## Steps

1. Capture the diff the PR will actually contain — the whole branch against the
   base it targets, not the working tree:
   ```bash
   git diff <base>...HEAD          # <base> is the branch the PR targets
   ```
   Use three dots: it diffs from the merge base, so commits that landed on
   `<base>` after you branched do not show up as your changes. `<base>` is
   usually `main`, but for a stacked PR it is the branch below yours — diffing
   against `main` there would hand the reviewer every PR under you as well.

   If you still have uncommitted work you want included, add it too, and note
   that a plain `git diff` sees neither staged nor untracked files:
   ```bash
   git add -N .                    # intent-to-add: new files become visible, stages no content
   git diff HEAD
   ```
   `git diff HEAD` covers staged and unstaged tracked files; `git add -N` is
   what makes a brand-new module or test visible at all, and it respects
   `.gitignore` so build artifacts stay out.
2. Read `.agents/knowledge/constraints.md` for known constraints.
3. **Launch a review subagent** with your agent's subagent/task mechanism (see
   the prompt below). The subagent receives only the diff + constraints — NOT
   your reasoning — to avoid confirmation bias. Point it at the diff command
   rather than pasting, so it reads the current state.
4. Act on the verdict.

| Verdict | Action |
|---------|--------|
| **safe** | Open or update the PR |
| **needs-attention** | Address the listed issues, then open or update the PR |
| **risky** | Output the report, do NOT open the PR, wait for the user |

5. Run `make quality` before pushing. (It gates every commit anyway, but this
   is the last chance before the diff is public.)

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
- Tests: does the diff extend an existing CI-enumerated test, or add a new file that the workflow owning that path actually lists? Check the owning workflow rather than assuming — `tests/data/` runs wholesale in both unit workflows, `tests/ops/` only in the GPU one (NPU enumerates ops files by name, so an Ascend-relevant ops file still needs a line), the e2e paths belong to `{gpu,npu}_e2e_test.yml`, and everything else must be listed file by file or it never runs. See `.agents/knowledge/testing.md`.
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
