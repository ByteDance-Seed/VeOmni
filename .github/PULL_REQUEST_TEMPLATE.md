### What does this PR do?

> Concise overview of the change. Reference related issues/PRs.

### Checklist Before Starting

- Search for relative PRs/issues and link here: ...
- PR title follows `[{modules}] {type}: {description}` format (see [check_pr_title.yml](.github/workflows/check_pr_title.yml) for the full list of allowed modules and types)
  - Breaking changes: prepend `[BREAKING]` — e.g. `[BREAKING][parallel, model] feat: dynamic batching`

### Test

> Validation results (training curves, eval metrics) for changes not covered by CI.

### API and Usage Example

> Show API changes and usage examples if applicable.

### Design & Code Changes

> High-level design description and specific change list.

### Checklist Before Submitting

- Read the [Contribute Guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md)
- Applied pre-commit checks
- Added/updated documentation
- If `tasks/` training scripts were moved or renamed: updated `docs/` examples and verified `python3 scripts/ci/check_doc_task_paths.py` passes (also enforced by the **Check doc task paths** CI workflow)
- Added tests to CI workflow (or explained why not feasible)
