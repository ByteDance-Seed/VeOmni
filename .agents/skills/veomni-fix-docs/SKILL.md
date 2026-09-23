---
name: veomni-fix-docs
description: Audit and correct VeOmni documentation links, Markdown, language, and technical claims against repository sources. Use for documentation scans, broken links, typos, or stale instructions in README, docs, examples, and agent guidance.
---

# Fix VeOmni Documentation

Audit documentation and apply only corrections supported by evidence. This skill is self-contained: it requires no bundled script, extra Python package, GPU, or network connection. Use the repository's available search, build, and validation tools; do not imply that a manual search is an exhaustive parser-based scan.

## Scope and baseline

1. Read the request and applicable `AGENTS.md`. A review-only request calls for findings, not edits. A correction request permits verified documentation fixes, not unrelated code changes or a push.
2. Run `git status --short` before editing. Preserve unrelated changes. Use requested paths; otherwise inspect tracked Markdown/MDX in `README.md`, `docs/`, `.agents/`, and component documentation. Include relevant new unignored files.
3. Record the repository revision and which files or directories you actually checked. For large scopes, work in batches and report any uninspected area instead of claiming full coverage.

## Mechanical checks

Enumerate candidates with `git ls-files --cached --others --exclude-standard -- '*.md' '*.mdx'`. Do not use `rg --files`, which skips hidden paths such as `.agents/`, or plain `git ls-files`, which omits untracked files. Search for Markdown links/images (including reference definitions), HTML `a[href]` and `img[src]`, headings, code fences, and repeated words. Search output is a candidate list, not proof that all links were found: multiline Markdown, footnotes, comments, code spans, MyST directives, and HTML can defeat simple patterns. Use an available Markdown parser or renderer when comprehensive coverage matters, without introducing a permanent dependency just for this skill.

For each candidate link or image:

- Resolve a relative path from the containing document, including `..`, URL decoding, and exact filename case. Do not resolve every link from the repository root. Check whether an absolute-site path is handled by the documentation renderer.
- For a fragment, inspect the target heading or explicit HTML ID and the **intended renderer's** generated permalink. GitHub and Sphinx/MyST can produce different IDs, especially for Emoji, linked heading text, punctuation, and duplicate headings. Do not remove an icon or change a heading merely to make a guessed fragment match.
- Check reference-style links against their definitions. Exclude literal code and commented examples unless the user asks to review examples.
- Treat malformed URLs, suspicious duplicates, missing spaces after heading markers, and unmatched fences as review candidates. Verify context before editing. Do not guess a replacement URL or destination.
- Check external links only if network access is available and authorized; otherwise mark them unverified. MyST labels/directives, generated targets, MDX/JSX routing, and site-specific URL rules need renderer-specific checks.

If a target is missing or ambiguous, search the repository for its intended replacement. Fix only when the destination is clear; otherwise report the uncertainty with file and line.

## Meaning and language

Review prose outside code blocks for objective spelling, grammar, punctuation, incomplete sentences, contradictory guidance, and outdated terminology. Verify commands, working directories, CLI flags, configuration paths and keys, defaults, versions, hardware claims, model registration examples, and API names against implementation, tests, current configs, CI, or Dockerfiles. `rg` definitions and call sites rather than relying on nearby prose. For procedures, check prerequisites and ordering; for examples, preserve executable quoting, indentation, and placeholders.

Apply a technical correction when an authoritative repository source proves it. Apply a language correction when there is only one reasonable reading. If evidence is insufficient, retain the original and report the candidate with the source checked and what remains uncertain. Do not turn a completed link check into a claim of semantic correctness.

## Edit and validate

1. Keep edits minimal. Preserve document language, voice, heading structure, Emoji, intentional hard breaks, and Markdown formatting. Change code blocks or commands only with evidence. Update directly affected cross-references.
2. Recheck each changed link and claim against its target or source. Run available repository checks, such as `python scripts/ci/check_doc_task_paths.py` for task paths and `python scripts/ci/check_agent_doc_paths.py` for agent paths. When documentation dependencies are available, build Sphinx HTML and inspect warnings; for changed anchors, compare rendered links with actual generated IDs. A successful build alone does not prove every anchor is correct.
3. Run applicable formatting/quality checks, `git diff --check`, and inspect the final diff. Do not install heavy training dependencies solely for documentation review.
4. Report the actual scope, fixes and supporting evidence, checks run, false positives, and unverified areas (especially external URLs, MyST/MDX constructs, and semantic claims). Never state that all documentation is error-free unless that conclusion is genuinely supported.
