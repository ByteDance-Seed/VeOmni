---
name: seedomni-v2
description: "SeedOmni V2 development guidance for Open-VeOmni. Use when adding or modifying `veomni/models/seed_omni/`, Omni modules, module mixins, CPU preprocessors, conversation carrier handling, training or generation graph YAML, SeedOmni split-checkpoint scripts, per-module accelerators, distributed/eager Omni inference, or SeedOmni V2 validation. Triggers include: new OmniModule, modify SeedOmni module, wire graph, update `training_graph` or `generation_graph`, split SeedOmni checkpoint, add CPUPreprocessor, change conversation_list flow, configure per-module FSDP2/DDP/emb/ep, or debug SeedOmni V2 train/infer graph behavior."
---

# SeedOmni V2 Skill

Use this file as a routing entrypoint. Load only the references needed for the
task at hand; do not read every reference by default.

## First Checks

1. Confirm the task touches SeedOmni V2:
   - Source code under `veomni/models/seed_omni/`.
   - Configs under `configs/seed_omni/`.
   - SeedOmni data collation or `conversation_list` preprocessing.
   - SeedOmni checkpoint splitting or module registration.
2. Treat current code as the source of truth. Some older docs still contain
   stale examples; prefer graph/module source and Janus configs when they
   disagree.
3. Choose the smallest applicable reference set from the routing table below.
4. Prefer live examples over synthetic examples. Use Janus and Qwen modules as
   implementation maps; avoid inventing placeholder module code.

## Routing

| Task | Read |
|---|---|
| Understand current authoritative files | `references/source-of-truth.md` |
| Change data flow, `conversation_list`, CPU preprocessing, dummy items, or raw request preprocessing | `references/data-flow.md` |
| Add or modify a module mixin/model/config/processor | `references/module-contract.md`, `references/module-workflow.md`, `references/janus-example-map.md` |
| Edit training graph or generation FSM YAML | `references/graph-runtime.md`, then the matching `templates/*.yaml` |
| Add a new model config layout | `references/graph-runtime.md`, `templates/base.template.yaml`, `templates/modules_train.template.yaml` |
| Add distributed/eager inference module config | `references/per-module-parallel.md`, `templates/modules_infer_eager.template.yaml`, `templates/modules_infer_fsdp.template.yaml` |
| Update split-checkpoint conversion | `references/checkpoint-splitting.md`, `references/janus-example-map.md` |
| Validate a SeedOmni V2 change | `references/validation.md` |

## Current Baseline

- Entry-time samples use `{"conversation_list": [...]}`. The SeedOmni collator
  groups samples and runs ordered module CPU preprocessors before GPU forward.
- `CPUPreprocessor` is a core data-flow contract, shared by training and
  inference. Training runs it inside `SeedOmniCollator`; inference runs it once
  before the FSM.
- Modules communicate by mutating/returning the shared `conversation_list`
  carrier, not by hidden edge payloads.
- Training graphs are flat edge lists over `module[.method]` endpoints.
  `to: end` is the virtual sink. Execution order is derived by topo sort.
- Generation graphs are FSMs. Each state body is an ordered inline edge list.
  `done` is framework-injected; use `module_signal` and `default` transitions.
- Tokenizers, processors, and other assets belong to the owning module
  subfolder. Do not add top-level tokenizer paths.

## Templates

Templates under `templates/` are skeletons, not authoritative examples. Use them
to start a new config file, then compare against the live Janus configs listed
in `references/janus-example-map.md`.

## Sibling Skills

- Use `veomni-new-model` first when adding a new HuggingFace model family under
  `veomni/models/transformers/`.
- Use this skill after that model exists, when wrapping it as SeedOmni V2
  modules or wiring it into Omni graphs.
- Use `veomni-develop` for unrelated framework work outside SeedOmni V2.
