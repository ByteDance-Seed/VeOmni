Lightweight bugfix for straightforward bugs. For complex/unclear bugs, use `veomni-debug`.

| Situation | Use |
|-----------|-----|
| Clear error, obvious fix | `veomni-bugfix` |
| Fix in <30 min | `veomni-bugfix` |
| Root cause unclear | `veomni-debug` |
| Multiple failed attempts | `veomni-debug` |
| Distributed training issues | `veomni-debug` |
| Numerical accuracy / loss divergence | `veomni-debug` |

## Steps

### 1. Reproduce
Get exact command and config. Confirm the error is consistent. Can't reproduce -> `veomni-debug`.

### 2. Identify Root Cause
- Read error/traceback carefully.
- Check `.agents/knowledge/constraints.md` for known pitfalls.
- Search for similar patterns in existing tests under `tests/`.
- Not clear in 15 min -> `veomni-debug`.

### 3. Fix
- Write a reproducer test if feasible (add to `tests/`).
- Minimal fix for root cause — don't "fix" surrounding code.
- Ensure fix works for all affected modalities (text, VLM, DiT) if the bug is in shared code.

### 4. Verify
- Reproducer test passes.
- Run `pytest tests/` for the affected module to check for regressions.
- Fix addresses root cause, not symptoms.

### 5. Commit
- Run `veomni-review` skill.
- Run `make quality` to ensure ruff compliance.
- PR title format: `[{module}] fix: {description}`.
