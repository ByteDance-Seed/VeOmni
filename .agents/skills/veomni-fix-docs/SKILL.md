---
name: veomni-fix-docs
description: "Scan, verify, and correct VeOmni documentation errors with deterministic checks and source-backed semantic review. Use for documentation audits, broken links, stale commands or paths, Markdown syntax, grammar, punctuation, unclear wording, and documentation fixes in README, docs, examples, or agent guidance. Trigger: 'scan docs', 'fix documentation', 'broken link', 'documentation typo', 'proofread VeOmni', or 'check docs against code'."
---

# Fix VeOmni Documentation

Use a two-pass workflow: detect mechanical defects automatically, then verify language and technical meaning against repository sources. Make only evidence-backed edits.

## 1. Establish scope and baseline

1. Run `git status --short` and preserve unrelated worktree changes.
2. Treat tracked and new unignored Markdown as the default scope. Narrow the scope only when the request names specific files.
3. Read the nearest `AGENTS.md` and the documentation source files relevant to the subject.

## 2. Run deterministic checks

From the repository root, run:

```bash
python .agents/skills/veomni-fix-docs/scripts/scan_docs.py --root .
python scripts/ci/check_doc_task_paths.py
```

The scanner reports broken local links and images, missing Markdown heading spaces, unclosed fenced code blocks, and adjacent duplicate words. Pass file or directory paths after the options to scan a subset. Use `--format json` for machine-readable output and `--fail-on warning` when warnings must fail CI.

Treat every report as a candidate, not permission to edit automatically. Duplicate words can be intentional; links containing templates or generated placeholders may require context.

## 3. Review meaning and language

Read [references/review-checklist.md](references/review-checklist.md) before the semantic pass. Review prose outside code blocks for:

- grammar, spelling, punctuation, and sentence completeness;
- commands, paths, API names, configuration keys, defaults, and version claims;
- contradictions between documents or with the current implementation;
- ambiguous prerequisites, unsupported guarantees, and stale terminology.

Use `rg` to find definitions and call sites. Prefer implementation, tests, current configs, and CI workflows over neighboring prose as evidence. If a statement cannot be verified locally, report it instead of guessing.

## 4. Apply minimal corrections

1. Preserve the document's language, voice, heading structure, and intentional Markdown formatting.
2. Keep code blocks and commands unchanged unless repository evidence proves they are wrong.
3. Separate factual corrections from optional style rewrites; omit purely subjective rewrites.
4. Update all directly affected cross-references when a path, command, or term changes.

## 5. Validate and report

1. Re-run the scanner on every changed document, then on the full default scope.
2. Re-run `python scripts/ci/check_doc_task_paths.py` when task paths appear in scope.
3. When the documentation dependencies are available, run `python -m sphinx -b html --keep-going docs docs/_build/html` and review warnings against the baseline.
4. Run the repository's applicable formatting or quality checks.
5. Inspect `git diff --check` and the final diff.
6. Report fixed findings, intentionally rejected candidates, verification evidence, and any checks unavailable in the environment.

Do not claim that the documentation is error-free. State the scanned scope and checks performed.
