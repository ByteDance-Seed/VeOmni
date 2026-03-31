---
name: veomni-refactor
description: "Safe refactoring: same behavior, better structure. If behavior changes, use veomni-feature or veomni-bugfix instead. Trigger: 'refactor', 'reorganize', 'clean up', 'restructure', 'move code', 'rename'."
---

## Phase 1: Baseline

1. Run tests before changes — record as baseline (`pytest tests/`).
2. Identify scope and why the refactor is needed.
3. Search all callers/importers of the code being refactored. Key areas to check:
   - `tasks/` entry points that may import from the module
   - `veomni/trainer/` subclasses that override methods
   - `veomni/models/` auto-registration patterns
   - `tests/` that directly test the module
   - `configs/` that reference class names or config keys

## Phase 2: Incremental Refactor

For each step:
1. ONE structural change.
2. Update ALL callers and importers.
3. Run tests — must match baseline.
4. Commit immediately.

**Never batch multiple refactoring steps into one commit.**

## Phase 3: Verify

1. Run same tests as Phase 1 — results must be identical.
2. Run `/veomni-review` skill on full diff.
3. Run `make quality` to ensure ruff passes.

## Common Traps

- `veomni.models.auto` registration depends on import-time side effects — moving registrations can break model loading.
- Renaming config keys breaks existing YAML configs in `configs/` — check all YAML files.
- Changing `BaseTrainer` method signatures breaks all subclasses (`TextTrainer`, `VLMTrainer`, `DitTrainer`, etc.).
- `veomni.distributed` modules are used differently by FSDP vs FSDP2 paths — test both if touching shared code.
- Data collators (`veomni/data/data_collator.py`) are tightly coupled to model-specific preprocessing — verify cross-modality compatibility.
