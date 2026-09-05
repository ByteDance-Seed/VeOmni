# Documentation Review Checklist

Use this checklist after the deterministic scanner. Consult only the sections relevant to the files under review.

## Technical accuracy

- Resolve every command, script path, config path, and referenced file from the repository root or from the working directory stated by the document.
- Verify CLI flags and configuration keys at their parser or dataclass definitions.
- Verify defaults in code or committed configs; do not infer them from another guide.
- Confirm class, function, module, model, backend, and environment-variable names with `rg`.
- Check version, platform, and hardware claims against `pyproject.toml`, lock files, CI workflows, Dockerfiles, and guarded code paths.
- Check ordered procedures for missing prerequisites and impossible sequencing.
- Treat examples as executable interfaces: preserve quoting, indentation, continuations, and placeholder boundaries.

## Cross-document consistency

- Search for renamed or corrected terms across `README.md`, `docs/`, `configs/`, `docker/`, `.agents/`, and component READMEs.
- Confirm that a moved script or config has no stale path references.
- Check that link labels still describe their destinations.
- Prefer one canonical term for the same concept, while preserving official third-party names.

## Language and punctuation

- Fix objective spelling, agreement, tense, article, capitalization, and punctuation errors.
- Split run-on sentences and complete fragments only when meaning is unambiguous.
- Replace ambiguous pronouns or modifiers when the intended referent is evident.
- Preserve established project terminology, acronyms, and the language used by the surrounding section.
- Avoid broad stylistic rewrites, marketing embellishment, and changes based only on personal preference.

## Markdown integrity

- Keep heading levels hierarchical and add a space after heading markers.
- Balance fenced code blocks and preserve their language tags.
- Resolve relative links from the containing file, not from the repository root, unless the path begins with `/`.
- Verify image paths and case sensitivity for Linux readers.
- Include HTML `img` and `a` elements when checking embedded assets and links.
- For heading links, compare the destination with the intended renderer's actual permalink. Formatting, linked heading text, Emoji, and duplicate titles can change generated IDs. Do not infer an anchor from the raw heading alone.
- On Sphinx/MyST pages, check the generated HTML for affected anchors and MyST labels/directives; the standalone scanner does not validate these renderer-specific constructs.
- Distinguish intentional Markdown hard breaks (two trailing spaces) from accidental whitespace.
- Leave code spans, URLs, generated output, and quoted error messages verbatim unless they are themselves the documented defect.

## Evidence threshold

Apply a correction when at least one authoritative repository source proves it, or when the language defect has only one reasonable correction. Otherwise, record the candidate with the evidence needed to decide it.

For each uncertain technical claim, record the document location, the source checked, and what remains unverified. A missing tool or insufficient evidence is a reason to retain that claim for review, not to guess a replacement.
