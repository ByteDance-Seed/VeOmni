---
name: pr-draft-maintainer
description: Maintain pull request or merge request descriptions and local PR draft markdown across repositories. Use when updating PR text from reports, benchmark results, CI or test logs, reviewer feedback, issue context, changelogs, release notes, or when syncing a local draft with a hosted PR via GitHub CLI or similar tools; for PR body and title refinement without changing source code.
---

# PR Draft Maintainer

## Scope

Maintain pull request prose. Do not modify implementation files, tests, configs, branch history, or commits unless the user explicitly asks. Work with any repository convention: hosted PR only, local markdown draft, title/body split, template-based body, issue-tracker text, GitHub PR, or GitLab merge request.

## Core rules

- Discover the repo's convention before editing: PR template, contribution guide, existing draft files, prior PRs, and hosted PR metadata.
- Preserve the existing title/body format unless the user asks to restructure it.
- Use only defensible facts from supplied artifacts or inspected state. Do not invent tests, benchmark numbers, approvals, or CI status.
- Separate evidence types clearly: functional tests, CI, performance, correctness parity, risk, rollout, known limits.
- Keep public PR text concise, reviewer-oriented, and in the same language as the existing PR/template unless the user requests another language.

## Workflow

1. **Resolve the target**
   - Identify the PR/MR URL, number, repository, branch, and hosting tool.
   - Identify local draft files if any. Search common paths only when needed: `.pr-drafts/`, `.github/`, `docs/`, `changelog`, or paths supplied by the user.
   - Identify source artifacts: reports, logs, benchmark output, screenshots, release notes, reviewer comments, issue links, or prior draft text.

2. **Load current context**
   - Read the existing local draft and source artifacts.
   - Read the repo PR template if present, for example `.github/PULL_REQUEST_TEMPLATE.md`.
   - If syncing or updating a hosted PR, inspect the current remote title/body first. GitHub example:
     ```bash
     gh pr view PR_NUMBER --json title,body,url,headRefName,baseRefName,isDraft
     ```
   - Check whether draft files are gitignored. If ignored, validate with `sed`, `grep`, `wc`, or `stat`; do not rely on `git diff`.

3. **Determine the draft format**
   - Body-only markdown.
   - First line is title, then blank line, then body.
   - Separate title and body files.
   - Hosted PR body with no local draft.
   - Repository-specific template sections.

   Preserve the detected format. If ambiguous, avoid putting a title line into the body unless the existing draft already does that.

4. **Integrate new material**
   - Add new facts to the most relevant existing section instead of appending a disconnected block.
   - For test logs: include command, outcome, and meaningful scope; omit noisy raw logs.
   - For reviewer feedback: summarize the addressed concern and the resulting change or clarification.
   - For issue context: link or mention issue IDs and explain the resolved user-visible problem.
   - For release notes or changelogs: keep user impact separate from implementation details.
   - For screenshots or demos: include only stable links/paths and a one-line explanation of what they verify.

5. **Handle performance data carefully**
   - Include workload, hardware/runtime, timing method, sample size, and warmup/exclusion policy when relevant.
   - Separate steady-state throughput claims from profiler counters or narrow trace-window observations.
   - State one-time costs separately from steady-state results.
   - Use compact tables for core metrics: latency, throughput, memory, CPU/GPU utilization, launch counts, error rate, or other domain-specific metrics.
   - Avoid broad production claims from a narrow benchmark. Say “in this tested scenario” and preserve caveats.

6. **Sync hosted PR only when requested or clearly implied**
   - Prefer file-based updates to avoid shell escaping issues.
   - GitHub body-only example:
     ```bash
     gh pr edit PR_NUMBER --body-file draft.md
     ```
   - GitHub title-plus-body draft example:
     ```bash
     title=$(sed -n '1p' draft.md)
     tail -n +3 draft.md > /tmp/pr-body.md
     gh pr edit PR_NUMBER --title "$title" --body-file /tmp/pr-body.md
     ```
   - If using GitLab or another host, use the equivalent CLI/API pattern and still read back the final remote text.
   - If the user only asks to draft locally, do not sync remotely.

7. **Validate the result**
   - Re-read the edited draft or hosted PR body.
   - Grep or print key updated lines and any changed tables.
   - Recompute simple deltas when source values allow it.
   - Confirm no unrelated files were modified.

## Quality bar

A good PR update answers, in order:

1. What changed?
2. Why was it needed?
3. How was it validated?
4. What changed in behavior, API, performance, or risk?
5. What should reviewers focus on?

Do not force all five headings into every PR. Match the repository template and keep the text as short as the evidence allows.

## Safety checks

- Do not fabricate validation, benchmark results, compatibility, approvals, or deployment status.
- Do not expose secrets, internal-only hostnames, credentials, or irrelevant local filesystem paths in public PR text.
- Do not run expensive tests or benchmarks for a text-only update unless explicitly asked.
- If important source artifacts conflict and three reconciliation attempts fail, stop and ask which source is authoritative.
