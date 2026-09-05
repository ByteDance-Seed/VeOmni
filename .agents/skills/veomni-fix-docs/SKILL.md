---
name: veomni-fix-docs
description: "Audit and correct VeOmni documentation: local links, heading anchors, Markdown, language, and claims checked against source code. Use for documentation scans, broken links, typos, or stale instructions in README, docs, examples, and agent guidance."
---

# Fix VeOmni Documentation

Run the mechanical checks, then verify language and technical meaning against repository sources. The scripts perform parsing and path checks; the model follows the decision table below. No specific model, editor, or subagent service is required by this skill.

## 1. Establish scope and baseline

1. Read the request and nearest `AGENTS.md`. For a review-only request, report findings; for a correction request, apply verified fixes.
2. Run `git status --short` and preserve unrelated changes.
3. Use the requested files/directories, or tracked and new unignored Markdown by default. Paths are relative to `--root`. Without Git, directory discovery excludes common build/environment folders but cannot apply `.gitignore`.

## 2. Set up and run the scanner

Use Python 3.11+ in the active environment. From the repository root, install the small scanner dependencies once, then run:

```bash
python -m pip install -r .agents/skills/veomni-fix-docs/requirements.txt
python .agents/skills/veomni-fix-docs/scripts/scan_docs.py --root . --format json --fail-on warning
```

These dependencies are separate from VeOmni training dependencies; scanning needs no GPU, model API, or network connection. Use the same Python interpreter for installation and execution. If the folder was installed elsewhere, substitute its actual path. To scan a subset, append paths such as `docs/key_features/model_loader.md`.

Read the result as follows:

- Exit **0**: no findings at the selected failure threshold; it does not prove all documentation is correct.
- Exit **1**: findings were produced. Review them using the table below; this is not an execution failure.
- Exit **2**: setup or scope error. Read `error`, correct the interpreter/dependencies/path, and retry. If tools or dependencies are unavailable, record the check as not run.

Each finding has `path`, `line`, `severity`, `rule`, and `message`. `coverage.skipped` counts skipped targets/constructs; `coverage.not_checked` lists remaining limits. Keep these limits in the final report. Omit `--format json` for human-readable output. `--fail-on error` permits review warnings; `--fail-on never` only suppresses finding-related failure codes.

## 3. Handle findings

| Rule | Next action |
|---|---|
| `path-case-mismatch` | Compare with the actual path shown; use its exact filename and directory case if it is the intended target. |
| `broken-local-link` | Search the repository for the intended file. Correct the link only when the target is unambiguous; otherwise report it for confirmation. |
| `invalid-url` | Check the surrounding text and intended destination. Correct a demonstrated syntax mistake; do not invent a URL. |
| `missing-anchor` | Verify the rendered heading permalink or generated HTML before changing the link. Preserve the heading text and Emoji unless they are independently wrong. |
| `duplicate-word`, `heading-space`, `unclosed-fence` | These are review candidates, not proof of invalid Markdown. Check whether the repetition, literal text, or fence ending is intentional. |
| `encoding`, `read-error`, `unverified-anchor` | Report the inaccessible file/target. Do not rewrite its encoding or claim its links were verified. |

The scanner checks Markdown links/images, used reference links, HTML `a[href]`/`img[src]`, exact path case, GitHub-style heading anchors, and explicit HTML IDs. It respects code blocks, code spans, comments, and footnotes. A line number identifies the start of a link/tag or its containing inline block; multiline constructs can span later lines.

MyST directives/labels, generated anchors, MDX/JSX semantics, site-specific routing, external URL availability, and deeper language/technical correctness require separate verification. Do not edit an unfamiliar construct just to remove a scanner warning. For example, `## 📚 Overview` has the GitHub anchor `#-overview`; Sphinx may emit `#overview` after MyST resolves the source link. Check the intended renderer.

## 4. Review meaning and language

Read [references/review-checklist.md](references/review-checklist.md) before the semantic pass. Review prose outside code blocks for:

- grammar, spelling, punctuation, and sentence completeness;
- commands, paths, API names, configuration keys, defaults, and version claims;
- contradictions between documents or with the current implementation;
- ambiguous prerequisites, unsupported guarantees, and stale terminology.

Use `rg` to find definitions and call sites. Prefer implementation, tests, current configs, and CI workflows over neighboring prose as evidence. For each technical edit, identify the source that supports it. When evidence is missing or the meaning cannot be established, keep the text and report it as unverified. Completing the mechanical scan does not require inventing a semantic correction.

## 5. Apply minimal corrections

1. Preserve the document's language, voice, heading structure, and intentional Markdown formatting.
2. Keep code blocks and commands unchanged unless repository evidence proves they are wrong.
3. Separate factual corrections from optional style rewrites; omit purely subjective rewrites.
4. Update all directly affected cross-references when a path, command, or term changes.

## 6. Validate and report

1. Re-run the scanner on every changed document, then on the full default scope.
2. Run `python scripts/ci/check_doc_task_paths.py` when task paths appear in scope and that VeOmni helper exists.
3. When the documentation dependencies are available, run `python -m sphinx -b html --keep-going docs docs/_build/html` and review warnings against the baseline. For anchor changes, compare the relevant generated link targets with actual HTML IDs; a successful build alone is insufficient.
4. Run the repository's applicable formatting or quality checks.
5. Inspect `git diff --check` and the final diff.
6. Report: scanned scope; confirmed fixes with evidence; intentional false positives; checks run and results; unverified findings and coverage limits.

When changing this skill's scanner, run its portable regression suite:

```bash
python -m unittest discover -s .agents/skills/veomni-fix-docs/tests -v
```

Do not claim that the documentation is error-free. State the scanned scope and checks performed.
