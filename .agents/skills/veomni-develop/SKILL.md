---
name: veomni-develop
description: "VeOmni-specific checklist for feature development and refactoring. Covers impact analysis across modalities, trainer hierarchy, data pipeline, and distributed code. Use before implementing any non-trivial change. For model-specific or ops-specific work, use veomni-new-model or veomni-new-op instead. Trigger: 'add feature', 'implement', 'refactor', 'reorganize', 'new capability'."
---

## Impact Analysis

Before implementing, check which areas your change affects:

| Area | What to check | Why it matters |
|------|--------------|----------------|
| `veomni/trainer/` | All trainer subclasses (`TextTrainer`, `VLMTrainer`, `DitTrainer`, RL trainers) | Changing `BaseTrainer` method signatures breaks all subclasses |
| `veomni/data/data_collator.py` | All modalities (text, VLM, DiT) | Collators are tightly coupled to model-specific preprocessing |
| `veomni/distributed/` | FSDP2 + ExtraParallel/MoE/SP paths | Shared distributed code is used by many downstream modalities |
| `veomni/models/auto.py`, `registry.py` | Model registry, import-time side effects | `MODELING_REGISTRY` is populated at import time; moving registrations breaks loading |
| `configs/` | YAML config keys | Renaming config keys breaks existing training configs silently |
| `veomni/models/transformers/*/` | `__init__.py` registration entry points | Model packages own patchgen configs and generated v5 modeling; edit the config and regenerate instead of editing generated files |

## Refactoring Safety Rules

When restructuring code (same behavior, better structure):

1. **Baseline first**: run `pytest tests/` before any change, record results.
2. **One change per commit**: ONE structural change → update ALL callers → verify tests match baseline → commit.
3. **Never batch multiple refactoring steps into one commit.**
4. Check baseline again at the end — results must be identical.

## Common Traps

- `veomni.models` registration depends on **import-time side effects** — moving registrations into functions or delaying them breaks model loading.
- Renaming config keys **silently breaks** existing YAML configs in `configs/` — grep all YAML files first.
- `veomni.distributed` modules feed into ExtraParallel/MoE/SP — touching shared code may affect every modality, so run the cross-cutting parallel tests.
- Data collators in `veomni/data/data_collator.py` are coupled to `DEFAULT_DATA_COLLATE_INFO` — adding new tensor keys requires updating the collate info table.
- `MainCollator` has **strict SP ordering** (pad → slice → FA kwargs → slice position_ids) — reordering breaks SP correctness.
- `position_ids == 0` marks segment boundaries for FA varlen — any transform that produces position_ids must preserve this convention.

## Documentation

Before committing, check if the change requires documentation updates:

- **New/changed API** → update or create docs in `docs/`.
- **New/changed extension point** (a mixin, hook, callback or base-class method that other modules or models must implement) → write a dedicated page, see below.
- **New/changed config fields** → update config examples in `configs/` and relevant docs.
- **Architecture change** → update `.agents/knowledge/architecture.md`.
- **New constraint discovered** → add to `.agents/knowledge/constraints.md`.

### Documenting an extension point

A contract that other code has to implement gets its own page in the same PR,
not a bullet in a usage doc. Mirror the source layout (SeedOmni mixins go to
`docs/seed_omni/mixins/<name>.md`) and add the page to a toctree in
`docs/index.md`: the docs build runs with `-W`, so a page outside every toctree
fails CI.

For each function in the contract, the page states:

1. **Who owns it**: the module must implement it, must call it, may override
   it, or the framework provides it and it must not be overridden.
2. **What it does**: inputs, return value, units, and edge cases (for example
   what it must return for empty input).
3. **Where the framework calls it**: the exact call site, not only the file
   that defines it.
4. **The end effect**: the metrics, outputs or behavior a user observes, and
   what happens for a module that does not opt in.

Also include a minimal opt-in example that matches the real module layout, the
call flow of one step or request as a nested list, the rules a correct
implementation must follow, and the tests that cover the contract.

Link every symbol and call site with a path relative to the doc plus a line
anchor, e.g.
`[metric_meter_add](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L120)`,
so the reader can jump to the code from the IDE and from GitHub. Prefer this
over `https://github.com/.../blob/main/...` URLs, which do not resolve until the
PR merges and do not open in the IDE. Keep call flows out of code fences: links
inside a fence are not clickable. Line anchors drift when code above them
changes, so re-check them after every code change in the PR (see
`/veomni-fix-docs`). Build with `make -C docs html SPHINXOPTS=-W` before
committing. Reference page: `docs/seed_omni/mixins/metric_meter.md`.

## Tests

Follow `.agents/knowledge/testing.md`. In short: extend an existing
CI-enumerated test before creating a new file. If you do create one, check its
path against the table there and wire it into the workflow that owns that path
— the directory-level entries (`tests/data/`, `tests/checkpoints/`, `tests/ops/` on GPU,
`tests/parallel/context_parallel/` on GPU) need no line, e2e paths belong to
`{gpu,npu}_e2e_test.yml`, and everything else is invisible to CI until it is
listed. A pure refactor with existing coverage does not need a new test — say
so in the PR instead of adding one.

## When to Use Other Skills

- **New model** → `/veomni-new-model`
- **New op/kernel** → `/veomni-new-op`
- **Bug fix or debugging** → `/veomni-debug`
- **Dependency update** → `/veomni-uv-update`
