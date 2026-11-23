---
name: create-pr
description: "Create a pull request for the current branch. Handles uncommitted changes, generates a PR title matching the `[{modules}] {type}: {description}` format enforced by CI, and fills in the PR description template. Trigger: 'create pr', 'open pr', 'submit pr', 'make pr'."
---

## Interaction Principle

**Minimize confirmations.** The only confirmation pause is for uncommitted changes (Step 1.3). Everything after that (drafting, writing `.pr-drafts/`, pushing, creating/updating the PR) runs automatically.

## Step 1: Pre-Flight Checks

1. Verify current branch (`git branch --show-current`) — if on `main`, stop and ask user to create a feature branch. Base branch defaults to `main` unless specified.
2. Check existing PR (`gh pr view --json number,title,body 2>/dev/null`).
3. Check uncommitted changes (`git status && git diff --stat`). If dirty, summarize and ask:
   > "There are uncommitted changes. Commit them before creating the PR?"
   - If yes: run `make quality` (stop on failure), stage relevant files (skip `.env`, credentials, binaries), commit.
   - If no: proceed with committed changes only.
4. Verify commits ahead of base (`git log origin/<base>..HEAD --oneline`; stop if empty).

## Step 2: Analyze Diff & Format Title

Inspect `git diff <base>...HEAD` and `git log <base>..HEAD --oneline`.

- **Title format**: `[{modules}] {type}: {description}` (prepend `[BREAKING]` if breaking; lowercase description, no period, <60 chars). Authoritative lists live in `.github/workflows/check_pr_title.yml`.
- **Module mapping**: `veomni/models/`→`model`, `veomni/trainer/`→`trainer`, `veomni/data/`→`data`, `veomni/distributed/`→`dist` (or `parallel` for strategy changes), `veomni/ops/`→`ops`, `veomni/checkpoint/`→`ckpt`, `veomni/optim/`→`optim`, `veomni/lora/`→`lora`, `configs/`→`config`, `docs/`→`docs`, `tests/`/`.github/workflows/`→`ci`, `docker/`→`docker`, `tasks/`→`task`, `.agents/`→`agent`, speed/memory claims→add `perf`, omni/telemetry/release→`omni`/`logging`/`release`, other→`misc`.
- **Types**: `feat`, `fix`, `refactor`, `chore`, `test`.

## Step 3: Review Gate & Draft File

1. **Run `/veomni-review`** over `<base>...HEAD` (per its applicability table). Stop and wait for the user if the verdict is `risky`.
2. **Read `.github/PULL_REQUEST_TEMPLATE.md`** directly (never from memory) and fill every section — especially `### Test`.
3. Write draft to `.pr-drafts/<pr-number>.md` (existing PR) or `.pr-drafts/<branch-name>.md` (new PR; rename after creation). Format: line 1 = PR title, line 2 = blank, line 3+ = body. Tell the user the draft path.

## Step 4: Push & Create / Update PR

```bash
git push -u origin HEAD
tail -n +3 .pr-drafts/<name>.md > /tmp/pr-body.md
# New PR:
gh pr create --base <base-branch> --title "<title>" --body-file /tmp/pr-body.md
# Existing PR:
gh pr edit <pr-number> --title "<title>" --body-file /tmp/pr-body.md
```

Report the PR URL and draft path. Never force-push unless explicitly requested.
