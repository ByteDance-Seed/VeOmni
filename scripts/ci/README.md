# CI tooling

Run these commands from the repository root.

## Accelerator tests

GPU and NPU unit/e2e suites share one entry point:

```sh
uv run --frozen python -m scripts.ci.accelerator --backend cuda --suite unit --include-conditional
uv run --frozen python -m scripts.ci.accelerator --backend cuda --suite e2e
uv run --frozen python -m scripts.ci.accelerator --backend npu --suite unit
uv run --frozen python -m scripts.ci.accelerator --backend npu --suite e2e --serial
```

See the [accelerator README](accelerator/README.md) for suite selection,
launch plans, scheduling, isolation and result artifacts. The
[case scheduling contract](../../docs/design/ci_case_scheduling.md) defines the
test-author API, and the [testing guide](../../docs/testing.md) describes test
coverage and extension rules.

New tests outside existing collection scopes are registered in
`accelerator/suites/gpu.json` and/or `accelerator/suites/npu.json`. See
[Registering Tests in CI](../../docs/testing.md#registering-tests-in-ci) for
examples and when no manifest change is needed.

## Documentation checks

```sh
python scripts/ci/docs/check_doc_task_paths.py
python scripts/ci/docs/check_agent_doc_paths.py
```

The first check verifies task paths in documentation shell examples. The second
checks repository paths and skill references in agent documentation. Both use
the repository root by default and accept `--repo-root` for another checkout.

## Layout

- `docs/`: documentation validation scripts.
- `accelerator/`: shared GPU/NPU test tooling; see its
  [module layout](accelerator/README.md#module-layout).
- `tests/ci/`, at the repository root: CPU tests for CI tooling, collected by
  both accelerator suite manifests.
