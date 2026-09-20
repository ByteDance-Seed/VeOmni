# Maintaining VeOmni documentation

The public site is built from this directory with Sphinx and MyST Markdown.
The `latest` site follows `main`. Keep commands consistent with the code in the
same checkout, and state a revision explicitly for historical validation results.

## Choose the reader and the source

| Content | Maintain it here |
| --- | --- |
| Project overview and starting links | Root `README.md` |
| Contribution workflow | Root `CONTRIBUTING.md` |
| Installation and the first training run | `get_started/` |
| Routine training and data workflows | `usage/` |
| Model family overviews and recipes | `models/<family>/`; catalog in `models/index.md` |
| Feature usage | `key_features/` |
| Contributor architecture and extension contracts | `developer/` |
| Implementation rationale | `design/` |
| Historical upgrades | `migrations/` (older pages may keep their URLs) |
| Accelerator-specific setup and limits | `hardware_support/` |
| Agent-specific constraints and procedures | `.agents/` at the repository root |

Keep an explanation in one authoritative page and link to it from other
entry points. Commands should reference the checked-in configuration rather
than reproduce its entire YAML. Public development instructions should be
readable without an agent; agent workflows should link to those instructions.
Group navigation by reader task even when older files retain their paths.

## Add or update a page

1. Give the page one descriptive H1 and state its audience, purpose, and scope.
2. Add it to a `{toctree}` reachable from `index.md`. Place user workflows before
   implementation details; keep migration history separate from current guidance.
3. Use relative Markdown links between pages built by Sphinx. For Markdown
   outside the documentation tree, use a GitHub link: Sphinx cannot resolve
   those files as source documents. Repository code and configuration files
   can use relative links (Sphinx exposes them as downloads).
   Link to the canonical explanation instead of maintaining a second copy.
4. Keep images in `assets/`, grouped by topic when there are several. Give them
   useful alternative text.
5. Verify every command, configuration key, version, and repository path against
   the current source. Distinguish a provided recipe from a hardware-validated
   recipe; record the environment and revision for validation claims.

Model recipes use six H2 sections: model introduction; variants and recipes;
environment and data; launch training; recipe configuration; validation and next
steps. Add each recipe to its family overview and toctree, then link the model
name from `models/index.md`. Keep YAML paths and implementation details on the
recipe page rather than in the top-level model catalog.
Mark placeholders explicitly. Separate a short user procedure from lengthy
implementation explanations and link the two.

When a workflow changes, rewrite the affected steps. A warning above obsolete
copy-paste instructions is not a substitute for updating them. When moving a
page, update incoming links and retain an old-path landing page with a link to
the replacement. Preserve widely linked heading anchors or provide an explicit
compatibility target. Do not delete old URLs as part of a navigation-only change.

## Build the docs

Use Python 3.12, matching the `Check docs build` GitHub Actions workflow. The lock file pins the
complete dependency closure used by CI. Run these commands from the repository root.

```bash
# Install dependencies.
python -m pip install -r docs/requirements-lock.txt

# Build the docs with the same warnings-as-errors contract as CI.
make -C docs clean
make -C docs html SPHINXOPTS=-W

# Check repository references used by documentation and agent instructions.
python3 scripts/ci/check_doc_task_paths.py
python3 scripts/ci/check_agent_doc_paths.py

# Check root entry-point links, built HTML links/anchors, and navigation reachability.
python3 scripts/ci/check_docs_integrity.py

# CPU-only checker regressions (also run by the documentation CI job).
python -m unittest discover -s tests/special_sanity -p test_docs_integrity.py
```

## Open the docs with your browser

```bash
python -m http.server -d docs/_build/html/
```
Launch your browser and open localhost:8000.

Check the homepage, sidebar, edited pages, and their links. In the PR, report
the build/path checks separately from any training validation. No new training
test is needed for a documentation-only change.

## What CI checks

`check_docs.yml` builds with warnings as errors and runs the integrity checker
on the same checkout's Sphinx artifacts. It triggers for root entry points,
documentation, scripts, configs, tests, and framework changes.

- Root `README.md`, `CONTRIBUTING.md`, and this authoring guide: local link and
  image targets exist, including reference-style Markdown links and raw HTML.
- Rendered site: local page, download, asset, and fragment targets exist.
- Sphinx navigation: every discovered page is reachable from the root toctree,
  except pages explicitly marked `orphan` (for intentional compatibility pages).
- The existing task/agent checks validate their specialized path contracts.

The checker makes no network requests and does not execute training examples.
External URL availability, source-only heading fragments in root Markdown,
configuration semantics, and hardware validation still require review. Model
catalog links are checked like other local configuration references; listing a
configuration does not certify a hardware combination.

## Navigation and presentation

The site uses the pinned PyData Sphinx theme with a small header/sidebar override.
`conf.py` defines the top-level section tabs; keep `DOC_SECTIONS` aligned with the
root toctree. Each section owns its sidebar subtree, while the right-hand contents
list is generated from the current page. Presentation lives in `_templates/` and
`assets/css/veomni.css`; it does not require external fonts or a client-side build.

Model recipe sources live in `models/<family>/`. Published `examples/*.md` URLs
remain as orphan compatibility includes with relative-link rewriting, so they
render the maintained recipe rather than a second copy. Use an explicit anchor
when renaming a previously published heading. Test the old and new URLs.

After a layout change, check desktop and mobile widths, section switching, family
expansion, search, keyboard navigation, and both color modes in a browser.
