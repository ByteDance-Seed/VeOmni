---
name: create-pr
description: "Create a pull request for the current branch. Handles uncommitted changes, generates a PR title matching the `[{modules}] {type}: {description}` format enforced by CI, and fills in the PR description template. Trigger: 'create pr', 'open pr', 'submit pr', 'make pr'."
---

## Interaction Principle

**Minimize confirmations.** The only user confirmation is for uncommitted changes (Step 1.3). Everything after that — drafting, writing file, pushing, creating PR — runs automatically. Tool-level permission prompts (file write, git push) serve as implicit confirmation; do not add extra "are you sure?" pauses on top.

## Steps

### Step 1: Pre-flight Checks

1. **Identify current branch**:
   ```bash
   git branch --show-current
   ```
   If on `main` — stop and ask the user to create a feature branch first.

2. **Determine base branch**: use `main` unless the user specifies otherwise.

3. **Check for uncommitted changes** (only confirmation point):
   ```bash
   git status
   git diff --stat
   ```
   If there are staged or unstaged changes, show a summary and ask the user:
   > "There are uncommitted changes. Commit them before creating the PR?"

   - If yes: run `make quality`, stage relevant files (skip `.env`, credentials, large binaries), commit.
   - If no: proceed with what's already committed.
   - If `make quality` fails: stop and report errors.

4. **Check for existing PR** on this branch:
   ```bash
   gh pr view --json number,title,body 2>/dev/null
   ```
   Note the PR number if one exists.

5. **Check commits ahead of base**:
   ```bash
   git log origin/main..HEAD --oneline
   ```
   If the branch has no commits ahead of `main`, stop — nothing to open a PR for.

### Step 2: Analyze Changes (automatic)

1. Collect the full diff against the base branch:
   ```bash
   git diff main...HEAD
   git log main..HEAD --oneline
   ```

2. Identify **affected modules** by mapping changed file paths to allowed module names:

   | Path prefix | Module |
   |------------|--------|
   | `veomni/models/` | `model` |
   | `veomni/trainer/` | `trainer` |
   | `veomni/data/` | `data` |
   | `veomni/distributed/` | `dist` |
   | `veomni/parallel/` | `parallel` |
   | `veomni/ops/` | `ops` |
   | `veomni/checkpoint/` | `ckpt` |
   | `veomni/optim/` | `optim` |
   | `veomni/logging/` | `logging` |
   | `configs/` | `config` |
   | `docs/` | `docs` |
   | `tests/`, `.github/workflows/` | `ci` |
   | `docker/` | `docker` |
   | `tasks/` | `task` |
   | `veomni/omni/` | `omni` |
   | `.agents/` | `agent` |
   | other / mixed | `misc` |

   Allowed modules: `misc`, `ci`, `config`, `docs`, `data`, `dist`, `omni`, `logging`, `model`, `optim`, `ckpt`, `release`, `task`, `perf`, `ops`, `parallel`, `docker`, `trainer`, `agent`

3. Determine **change type**:

   | Type | When |
   |------|------|
   | `feat` | New functionality or capability |
   | `fix` | Bug fix |
   | `refactor` | Same behavior, better structure |
   | `chore` | Maintenance, cleanup, config changes |
   | `test` | Test-only changes |

### Step 3: Generate Draft File and Push (automatic)

1. Draft PR title in `[{modules}] {type}: {description}` format:
   - Multiple modules separated by comma: `[model, data] feat: ...`
   - Description: concise, lowercase start, no period, under 60 chars
   - Breaking changes: prepend `[BREAKING]`
   - Must pass the regex in `.github/workflows/check_pr_title.yml`

2. Draft PR description following `.github/PULL_REQUEST_TEMPLATE.md`.

3. **Write to `.pr-drafts/`** (already in `.gitignore`):
   ```bash
   mkdir -p .pr-drafts
   ```

   **Filename convention**:
   - Existing PR: `.pr-drafts/<pr-number>.md` (e.g. `.pr-drafts/123.md`)
   - New PR: `.pr-drafts/<branch-name>.md` — renamed to PR# after creation.

   **File format** — first line is the PR title, blank line, then the description body:

   ```markdown
   [model] feat: add support for Qwen4

   ### What does this PR do?

   > Summary here.

   ### Checklist Before Starting

   - Search for relative PRs/issues and link here: ...
   - PR title follows `[{modules}] {type}: {description}` format

   ### Test

   > Test description here.

   ### API and Usage Example

   > N/A

   ### Design & Code Changes

   > - Change 1
   > - Change 2

   ### Checklist Before Submitting

   - [ ] Read the [Contribute Guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md)
   - [ ] Applied pre-commit checks
   - [ ] Added/updated documentation
   - [ ] Added tests to CI workflow (or explained why not feasible)
   ```

4. Tell the user the draft file path (so they know where to find it if they want to review later).

### Step 4: Push and Create/Update PR (automatic, immediately after Step 3)

1. Push the branch:
   ```bash
   git push -u origin HEAD
   ```

2. **Create or update**:
   - New PR (use `--body-file` to avoid shell escaping issues):
     ```bash
     # Extract body from draft file (everything after the first blank line)
     tail -n +3 .pr-drafts/<branch-name>.md > /tmp/pr-body.md
     gh pr create --base <base-branch> --title "<title>" --body-file /tmp/pr-body.md
     ```
     After creation, rename the draft file from `<branch-name>.md` to `<pr-number>.md`.
   - Existing PR:
     ```bash
     tail -n +3 .pr-drafts/<pr-number>.md > /tmp/pr-body.md
     gh pr edit <pr-number> --title "<title>" --body-file /tmp/pr-body.md
     ```

3. Output the PR URL and the draft file path.

## Common Pitfalls

- **Title format is enforced by CI** — a malformed title will block the PR. Always validate against the allowed modules and types listed above.
- **Don't force-push** unless the user explicitly asks.
- **Check for sensitive files** before committing — skip `.env`, credentials, large binaries and warn the user.
