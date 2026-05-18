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
| `veomni/models/auto.py`, `loader.py` | Model registry, import-time side effects | `MODELING_REGISTRY` is populated at import time; moving registrations breaks loading |
| `configs/` | YAML config keys | Renaming config keys breaks existing training configs silently |
| `veomni/models/transformers/*/` | `__init__.py` registration entry points | All models ship a patchgen-generated v5 path under `generated/`; never import or call legacy `modeling_<m>.py` or `apply_veomni_<m>_patch()` (these no longer exist) |

## Refactoring Safety Rules

When restructuring code (same behavior, better structure):

1. **Baseline first**: run `pytest tests/` before any change, record results.
2. **One change per commit**: ONE structural change → update ALL callers → verify tests match baseline → commit.
3. **Never batch multiple refactoring steps into one commit.**
4. Check baseline again at the end — results must be identical.

## Common Traps

- `veomni.models.auto` registration depends on **import-time side effects** — moving registrations into functions or delaying them breaks model loading.
- Renaming config keys **silently breaks** existing YAML configs in `configs/` — grep all YAML files first.
- `veomni.distributed` modules feed into ExtraParallel/MoE/SP — touching shared code may affect every modality, so run the cross-cutting parallel tests.
- Data collators in `veomni/data/data_collator.py` are coupled to `DEFAULT_DATA_COLLATE_INFO` — adding new tensor keys requires updating the collate info table.
- `MainCollator` has **strict SP ordering** (pad → slice → FA kwargs → slice position_ids) — reordering breaks SP correctness.
- `position_ids == 0` marks segment boundaries for FA varlen — any transform that produces position_ids must preserve this convention.

## Documentation

Before committing, check if the change requires documentation updates:

- **New/changed API** → update or create docs in `docs/`.
- **New/changed config fields** → update config examples in `configs/` and relevant docs.
- **Architecture change** → update `.agents/knowledge/architecture.md`.
- **New constraint discovered** → add to `.agents/knowledge/constraints.md`.

## When to Use Other Skills

- **New model** → `/veomni-new-model`
- **New op/kernel** → `/veomni-new-op`
- **Bug fix or debugging** → `/veomni-debug`
- **Dependency update** → `/veomni-uv-update`
