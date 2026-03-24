---
name: veomni-run-test
description: "Use this skill to run tests for VeOmni. Covers unit tests, e2e tests, distributed tests, and specific module tests. Handles test selection, execution, and result recording. Trigger: 'run tests', 'test this', 'verify changes', 'check regressions', 'pytest'."
---

## Step 1: Determine Test Scope

Based on the changes made, select the appropriate test scope:

| Change Area | Test Command | Description |
|-------------|-------------|-------------|
| `veomni/models/` | `pytest tests/models/` | Model loading, patching, registry |
| `veomni/data/` | `pytest tests/data/` | Data pipeline, collators, transforms |
| `veomni/ops/` | `pytest tests/ops/` | Kernel ops, flash attention, fused ops |
| `veomni/distributed/` | `pytest tests/parallel/` | FSDP, sequence parallel, MoE |
| `veomni/checkpoint/` | `pytest tests/checkpoints/` | Checkpoint save/load |
| `veomni/utils/` | `pytest tests/utils/` | Utility functions |
| `veomni/trainer/` | `pytest tests/e2e/` | End-to-end training |
| Full regression | `pytest tests/` | All tests |

## Step 2: Pre-flight Checks

Before running tests:

1. Activate the virtual environment: `source .venv/bin/activate`
2. Run `make quality` to ensure code passes ruff.
3. Check that test configs exist in `tests/toy_config/` for the relevant models.

## Step 3: Execute Tests

```bash
source .venv/bin/activate

# Unit tests for specific module
pytest tests/<module>/ -v

# With specific test selection
pytest tests/<module>/test_<name>.py -v -k "<test_pattern>"

# E2e tests (require GPU)
pytest tests/e2e/ -v

# All tests
pytest tests/ -v
```

### Distributed Tests

Tests in `tests/parallel/` and `tests/e2e/` may require multiple GPUs. These use `torchrun` or custom launch utilities from `tests/tools/launch_utils.py`.

```bash
# Example: run ulysses parallel tests
pytest tests/parallel/ulysses/ -v
```

## Step 4: Record Results

After tests complete:

1. **Note the results**: total passed, failed, skipped, errors.
2. **If any failures**: record the failing test names and error messages.
3. **Regressions**: If a previously passing test now fails, **STOP** and investigate before proceeding. Use `veomni-debug` if the cause is not obvious.

## Step 5: Quality Gate

Before declaring tests complete:

- [ ] All relevant unit tests pass
- [ ] `make quality` passes (ruff check + format)
- [ ] No new test warnings that indicate real issues
- [ ] If changes affect distributed code: parallel tests pass (or are skipped with documented reason)
