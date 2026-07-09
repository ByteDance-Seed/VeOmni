---
name: veomni-patch-to-pr
description: "Apply an external diff/patch or report bundle to VeOmni and turn it into a clean pull request. Use when the user says apply a .diff/.patch, create a branch, write/update a PR from a report, push/submit, or package experimental changes for review. Reuses /veomni-develop, /veomni-debug, /veomni-review, /pr-draft-maintainer, and /create-pr instead of replacing them."
---

# VeOmni Patch to PR

## Purpose

Turn a supplied patch plus optional reports/logs/reviewer context into a reviewable VeOmni branch and PR without leaking scratch artifacts, local paths, wrong authors, or unsupported claims.

## Workflow

1. **Resolve intent and inputs**
   - Identify the target repo, base branch, patch files, report/log files, PR URL/number if any, and whether the user wants local draft only or remote PR update.
   - If the user only asks for PR prose, use `/pr-draft-maintainer` and do not touch code.
   - If the task is a bug/CI failure rather than patch packaging, use `/veomni-debug` or `gh-fix-ci` first, then return here for PR packaging.

2. **Prepare a safe workspace**
   - Check `git status --short`, current branch, upstream, and recent commits.
   - Never apply a patch on `main`. Create a feature branch or worktree with a descriptive name.
   - Record original `user.name` and `user.email`; before committing, ensure the author is the intended contributor and avoid accidental `bytedance`/local-machine authorship.
   - Read `.agents/knowledge/constraints.md`; for feature or refactor work also use `/veomni-develop`, for new ops use `/veomni-new-op`, and for patchgen/model work use `/veomni-migrate-transformers-v5`.

3. **Apply and inspect the patch**
   - Prefer `git apply --check <patch>` before applying. If it fails, inspect reject context and decide whether to use `git apply --3way` or manual conflict resolution.
   - After applying, run `git status --short` and `git diff --stat`.
   - Inspect for scratch artifacts before they enter the PR: one-off scripts, raw reports, benchmark dumps, logs, generated temp files, local absolute paths, and files explicitly excluded by the user.
   - Generated files under `veomni/models/transformers/*/generated/` must only change as part of a regenerated patchgen flow; otherwise stop and fix the source config instead.

4. **Integrate evidence into code, docs, and PR text**
   - Convert supplied reports into concise PR validation statements; do not commit large raw reports unless the user asks and the repo convention supports it.
   - Use `/pr-draft-maintainer` for PR body updates from benchmarks, CI logs, reviewer feedback, or test reports.
   - Keep performance claims scoped to the supplied workload, hardware/runtime, sample size, and caveats.
   - Remove irrelevant local filesystem paths from public PR text.

5. **Validate the branch**
   - Run the smallest meaningful test set first, then required quality gates such as `make quality`.
   - If tests are unavailable locally, state the exact command that could not be run and why.
   - Run `/veomni-review` before committing. For a risky verdict, do not commit until the issue is fixed or the user approves the risk.

6. **Commit, push, and open/update the PR**
   - Commit only related changes. Do not batch cleanup, unrelated refactors, and feature changes together.
   - Use a VeOmni-compliant PR title: `[{modules}] {type}: {description}`.
   - Use `/create-pr` for the final GitHub PR creation flow, or `gh pr edit --body-file` when updating an existing PR.
   - After push, read back the remote PR title/body and run `gh pr checks` or explain when checks are pending/unavailable.

## Final response checklist

Report:
- Branch name and PR URL or draft path.
- What patch/report inputs were used.
- What was intentionally excluded from the PR.
- Tests/quality/review commands run and their results.
- Any remaining CI, reviewer, or environment follow-up.
