### What does this PR do?

> Concise overview of the change. Reference related issues/PRs.

### Checklist Before Starting

- Search for relative PRs/issues and link here: ...
- PR title follows `[{modules}] {type}: {description}` format (enforced by [check_pr_title.yml](workflows/check_pr_title.yml))
  - **Allowed modules:** `agent`, `ci`, `ckpt`, `config`, `data`, `dist`, `docker`, `docs`, `logging`, `lora`, `misc`, `model`, `omni`, `optim`, `ops`, `parallel`, `perf`, `release`, `task`, `trainer`
  - **Allowed types:** `feat`, `fix`, `refactor`, `chore`, `test`
  - Breaking changes: prepend `[BREAKING]` — e.g. `[BREAKING][parallel, model] feat: dynamic batching`

### Test

> Validation results (training curves, eval metrics) for changes not covered by CI.
> State which of the three applies: extended an existing CI-selected test, added a new test covered by the applicable GPU/NPU suite manifest, or no test needed (with the reason). For a new file outside existing collection scopes, add a selection with `suite: "unit"` or `suite: "e2e"` to `scripts/ci/accelerator/suites/gpu.json` and/or `npu.json`.

### API and Usage Example

> Show API changes and usage examples if applicable.

### Design & Code Changes

> High-level design description and specific change list.

### Checklist Before Submitting

- Read the [Contribute Guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md)
- Applied pre-commit checks
- Added/updated documentation
- If `tasks/` training scripts were moved or renamed: updated `docs/` examples and verified `python3 scripts/ci/docs/check_doc_task_paths.py` passes (also enforced by the **Check doc task paths** CI workflow)
- Tests: extended an existing CI-selected test, **or** added a new test covered by the applicable suite manifest, **or** explained in **Test** why none is needed. New functions/parameters already included by a selection, and files inside collected directories, need no extra entry. Files outside those scopes must be added to the GPU/NPU manifests; do not add per-file pytest workflow steps. See [registration rules](../docs/testing.md#registering-tests-in-ci) and [.agents/knowledge/testing.md](../.agents/knowledge/testing.md).
