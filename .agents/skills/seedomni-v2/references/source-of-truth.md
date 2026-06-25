# Source Of Truth

Use this reference to decide which files to trust before editing SeedOmni V2.

## Highest Priority

- `veomni/models/seed_omni/mixins/modulemixin.py`
  - `ModuleMixin`, `CPUPreprocessor`, `pre_forward`, `post_forward`.
- `veomni/models/seed_omni/graphs/graph.py`
  - endpoint parsing and common graph schema.
- `veomni/models/seed_omni/graphs/training_graph.py`
  - training DAG execution and `conversation_list` merge semantics.
- `veomni/models/seed_omni/graphs/generation_graph.py`
  - inference FSM execution, permissive routing, transition behavior.
- `veomni/models/seed_omni/modeling_omni.py`
  - module construction, graph build, training/generation entrypoints.
- `veomni/data/data_collator.py`
  - `SeedOmniCollator` and ordered CPU preprocessor execution.
- `veomni/data/seed_omni/seedomni_transform.py`
  - current data transform that emits `conversation_list`.

## Live Examples

- `veomni/models/seed_omni/modules/janus/`
  - Best complete multi-module example.
- `configs/seed_omni/Janus/janus_1.3b/`
  - Best complete training and inference config example.
- `veomni/models/seed_omni/modules/qwen3*/`
  - Useful for text-only, MoE, and vision-language variants.

## Docs

- `docs/seed_omni/design.md`
  - Background and design rationale. Read for intent, but verify schema details
    against current graph source because some old examples may remain.
- `docs/seed_omni/omni_v2_refactor_migration.md`
  - Useful migration notes. The CPU preprocessor section reflects the current
    shared training/inference direction.
- `docs/seed_omni/example_models/janus.md`
  - Janus pipeline notes when present.

## Skill Resources

- `references/*.md`
  - Task-specific contracts and workflows.
- `templates/*.yaml`
  - Config skeletons. They are starting points only; compare with live configs
    before committing.
