# Source Of Truth

Use this reference to decide which files to trust before editing SeedOmni V2.

## Highest Priority

- `veomni/models/seed_omni/mixins/`
  - `base_mixin.py` — shared assets / hook registry.
  - `training_module_mixin.py` — `pre_forward`, `post_forward`, training dispatch.
  - `inference_module_mixin.py` — live `reset_*` / `finalize` hooks; its `pre_generate` / `post_generate` dispatchers have no call-site (both FSM drivers call endpoints directly).
  - `metric_meter_mixin.py` — optional per-module FLOPs meter.
- `veomni/models/seed_omni/modules/**/modeling.py`
  - native HF class + in-file `InferenceMixin` (`generate()` + FSM state).
- `veomni/models/seed_omni/modules/**/accelerated.py`
  - composable `TrainingMixin` / `VeOmniMixin` + IDE type stubs; see `references/modulemixin-ide-stubs.md`.
- `veomni/models/seed_omni/graphs/base.py`
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

- `docs/seed_omni/seed_omni_v2.md`
  - Authoritative architecture and developer guide. Start here for intent, but
    verify schema details against current graph source.
- `docs/seed_omni/av_video_design.md`
  - Decided-but-unimplemented design for audio-bearing video. Read only when
    adding an audio modality.
- `docs/seed_omni/omni_v2_refactor_migration.md`
  - Historical merge notes; its path map predates `accelerator/`. Read only for
    the CPU preprocessor rationale, and verify every path against the tree.
- `docs/seed_omni/example_models/janus.md`
  - Janus pipeline notes when present.

## Skill Resources

- `references/*.md`
  - Task-specific contracts and workflows.
- `templates/*.yaml`
  - Config skeletons. They are starting points only; compare with live configs
    before committing.
