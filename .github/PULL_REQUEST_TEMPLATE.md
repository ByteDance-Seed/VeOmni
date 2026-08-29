### What does this PR do?

> Concise overview of the change. Reference related issues/PRs.

### Checklist Before Starting

- Search for relative PRs/issues and link here: ...
- PR title follows `[{modules}] {type}: {description}` format (enforced by [check_pr_title.yml](.github/workflows/check_pr_title.yml))
  - **Allowed modules:** `agent`, `ci`, `ckpt`, `config`, `data`, `dist`, `docker`, `docs`, `logging`, `lora`, `misc`, `model`, `omni`, `optim`, `ops`, `parallel`, `perf`, `release`, `task`, `trainer`
  - **Allowed types:** `feat`, `fix`, `refactor`, `chore`, `test`
  - Breaking changes: prepend `[BREAKING]` — e.g. `[BREAKING][parallel, model] feat: dynamic batching`

### Test

> Validation results (training curves, eval metrics) for changes not covered by CI.
> State which of the three applies: extended an existing CI-enumerated test, added a new test and wired it into the unit-test workflows, or no test needed (with the reason).

### API and Usage Example

> Show API changes and usage examples if applicable.

### Design & Code Changes

> High-level design description and specific change list.

### Checklist Before Submitting

- Read the [Contribute Guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md)
- Applied pre-commit checks
- Added/updated documentation
- If `tasks/` training scripts were moved or renamed: updated `docs/` examples and verified `python3 scripts/ci/check_doc_task_paths.py` passes (also enforced by the **Check doc task paths** CI workflow)
- Tests: extended an existing CI-enumerated test, **or** added a new test and wired it into `gpu_unit_tests.yml` (and `npu_unit_tests.yml` where applicable), **or** explained in **Test** why none is needed. Note that CI enumerates test files individually — a new file outside `tests/data/` and `tests/ops/` does not run until it is listed. See [.agents/knowledge/testing.md](.agents/knowledge/testing.md).
