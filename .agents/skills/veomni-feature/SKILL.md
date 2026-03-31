---
name: veomni-feature
description: "Structured workflow for developing a new feature in VeOmni. For non-model work: new trainer capabilities, data pipeline changes, distributed strategy improvements, new ops/kernels, config enhancements. Trigger: 'add feature', 'implement', 'new capability', 'support for', 'enhance'."
---

## Phase 1: Understand & Scope

1. **Clarify requirements**: What should change? What should NOT?
2. **Find related code**: Search codebase for similar patterns. Check `veomni/` modules.
3. **Identify affected areas**: List files/modules and their dependents. Pay attention to:
   - Trainer subclasses that may need updates (`veomni/trainer/`)
   - Config files that may need new fields (`configs/`)
   - Data collators/transforms that interact with the change (`veomni/data/`)
   - Distributed code with sharding assumptions (`veomni/distributed/`)
4. **Check constraints**: Read `.agents/knowledge/constraints.md`.

## Phase 2: Plan

1. Write design in TodoWrite: goal, approach, files, risks.
2. If touching >5 files or changing a public API, **wait for user review**.

## Phase 3: Implement

1. **Tests first** (when feasible) — add to `tests/` matching the module structure.
2. **Incremental changes**: one logical change at a time, commit after each.
3. **Follow existing patterns** — match the codebase style (ruff-compliant, English comments).
4. **Run tests** after each change: `pytest tests/<relevant_module>/`.

## Phase 4: Validate

1. Run the relevant test suite (see `/veomni-run-test` skill).
2. If the feature affects distributed training, verify with multi-GPU e2e tests.
3. Check side effects on other trainers/modalities.

## Phase 5: Finalize

1. Run `/veomni-review` skill (pre-commit code review).
2. Run `make quality` to ensure ruff passes.
3. Update documentation in `docs/` if the feature introduces new APIs.

## When to Use Other Skills

- **New model** -> `/veomni-new-model`
- **Bug fix** -> `/veomni-bugfix` or `/veomni-debug`
- **Refactoring** -> `/veomni-refactor`
- **New op/kernel** -> `/veomni-new-op`
