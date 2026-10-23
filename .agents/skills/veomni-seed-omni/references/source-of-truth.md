# Source Of Truth

Use this reference to decide which files to trust before editing SeedOmni.

## Highest Priority

- `veomni/models/seed_omni/mixins/`
  - `base_mixin.py` — shared assets / hook registry.
  - `training_module_mixin.py` — `pre_forward`, `post_forward`, training dispatch.
  - `inference_module_mixin.py` — live `reset_*` / `finalize` hooks; its `pre_generate` / `post_generate` dispatchers have no call-site (both FSM drivers call endpoints directly).
  - `metric_meter_mixin.py` — optional per-module FLOPs meter.
- `veomni/models/seed_omni/modules/**/modeling.py`
  - native HF class + in-file `InferenceMixin` (`generate()` + FSM state).
- `veomni/models/seed_omni/modules/**/accelerated.py`
  - composable `TrainingMixin` / `VeOmniMixin` + IDE type stubs; see `references/module-contract.md`.
- `veomni/models/seed_omni/graphs/base.py`
  - endpoint parsing and common graph schema.
- `veomni/models/seed_omni/graphs/training_graph.py`
  - training DAG execution and `conversation_list` merge semantics.
- `veomni/models/seed_omni/graphs/generation_graph.py`
  - inference FSM execution, permissive routing, transition behavior.
- `veomni/models/seed_omni/modeling_omni.py`
  - module construction, graph build, training/generation entrypoints.
- `veomni/data/seed_omni/seedomni_transform.py`
  - data transform that builds `conversation_list` and runs ordered CPU preprocessors (`OmniProcessor`).
- `veomni/data/seed_omni/collator.py`
  - `SeedOmniCollator` (pure list-grouping collator).

## Live Examples

- `veomni/models/seed_omni/modules/fake_model/` and `configs/seed_omni/fake_model/`
  - In-tree minimal multi-module stand-in.
- `veomni/models/seed_omni/modules/<family>/` and `configs/seed_omni/<Family>/<variant>/`
  - Complete multi-module family implementations (e.g. Janus, Qwen3, Qwen3-VL, BAGEL).

## Docs

- `docs/seed_omni/design/architecture.md`
  - Architecture: module split, conversation carrier, graph views, train and
    generation flow, file map. Start here for intent, but verify schema details
    against current graph source.
- `docs/seed_omni/distributed/per_module_parallelism.md`
  - What each module may choose (FSDP2, DDP, eager, `emb` / `ep` groups,
    frozen) versus what the job shares, per-module `ParallelState`, clipping.
- `docs/seed_omni/distributed/sequence_parallel.md`
  - Ulysses contract, why SP is one size for the whole graph, correctness
    invariants.
- `docs/seed_omni/design/media.md`
  - Video / audio reading and saving, plus the decided-but-unimplemented
    audio-bearing video design. Read when touching media or adding audio.
- `docs/seed_omni/usage/`
  - `data_format.md`, `training_and_inference.md`, `adding_a_model.md`, `hf_layout.md`.

## Skill Resources

- `references/*.md`
  - Task-specific contracts and workflows.
- `templates/*.yaml`
  - Config skeletons. They are starting points only; compare with live configs
    before committing.
