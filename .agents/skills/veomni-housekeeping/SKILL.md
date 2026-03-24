---
name: veomni-housekeeping
description: "Post-compaction housekeeping — verify state, sync understanding, resume work. Trigger after context compaction or when resuming a long session. This skill is an AUDITOR — it verifies state from file evidence, not from memory."
---

Context was just compacted — your working memory may be incomplete. Perform ALL steps below before resuming work.

**Cardinal rule: Do NOT fabricate or reconstruct information you don't remember.** Verify from file evidence only.

## 1. Verify Current State

Check the current state of the repository:

```bash
git status
git log --oneline -10
git diff --stat
```

- **Uncommitted changes?** Review them — they may be work in progress from before compaction.
- **Recent commits?** Read their messages to understand what was just completed.

## 2. Re-read Project Context

Re-read key files to restore context:

1. `.agents/knowledge/constraints.md` — hard constraints to keep in mind
2. `.agents/knowledge/architecture.md` — module structure overview
3. Any recently modified files (from `git diff --stat`)

## 3. Sync Todo

If using TodoWrite, cross-check against file evidence:

- `git log --oneline -5` — any commits since last known state?
- `git diff --stat` — any uncommitted work in progress?
- Mark items as completed only if file evidence confirms it.
- Add newly discovered items only if file evidence shows them.

## 4. Quality Check

Run a quick quality check to ensure the codebase is in a good state:

```bash
make quality
```

If there are ruff errors, fix them before resuming other work.

## 5. Resume

- If the user gave an explicit instruction before compaction -> follow that instruction.
- If there are uncommitted changes -> review and decide whether to commit or continue.
- If TodoWrite has pending items -> resume from the next pending item.
- Do NOT ask the user "what should I do next" if the todo list has a clear next step.
