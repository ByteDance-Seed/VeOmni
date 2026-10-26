---
name: veomni-seed-omni
description: "SeedOmni development guidance for Open-VeOmni. Use when adding or modifying `veomni/models/seed_omni/`, Omni modules, module mixins, CPU preprocessors, conversation carrier handling, training or generation graph YAML, SeedOmni split-checkpoint scripts, per-module accelerators, distributed/eager Omni inference, or SeedOmni validation. Triggers include: new OmniModule, modify SeedOmni module, wire graph, update `training_graph` or `generation_graph`, split SeedOmni checkpoint, add Preprocessor, change conversation_list flow, configure per-module FSDP2/DDP/emb/ep, or debug SeedOmni train/infer graph behavior."
---

# SeedOmni Skill

Use this file as a routing entrypoint and load the references matching the task
from the routing table below.

## First Checks

1. Confirm the task touches SeedOmni:
   - Source code under `veomni/models/seed_omni/`.
   - Configs under `configs/seed_omni/`.
   - SeedOmni data collation or `conversation_list` preprocessing.
   - SeedOmni checkpoint splitting or module registration.
2. Treat current code as the source of truth (graph/module source and live
   configs).
3. Choose the smallest applicable reference set from the routing table below.
4. Use live `fake_model`, Janus, and Qwen modules as implementation maps.

## Routing

| Task | Read |
|---|---|
| Understand SeedOmni architecture, runtime hierarchy, and hard constraints | `.agents/knowledge/veomni-seed-omni.md`, `references/source-of-truth.md` |
| Change data flow, `conversation_list`, CPU preprocessing, dummy items, or raw request preprocessing | `references/data-flow.md` |
| Add or modify a module mixin/model/config/processor | `references/module-contract.md` |
| Edit training graph or generation FSM YAML | `references/graph-runtime.md`, then the matching `templates/*.yaml` |
| Add a new model config layout | `references/graph-runtime.md`, `templates/base.template.yaml`, `templates/modules_train.template.yaml` |
| Add distributed/eager inference module config | `references/per-module-parallel.md`, `templates/modules_infer_eager.template.yaml`, `templates/modules_infer_fsdp.template.yaml` |
| Update split-checkpoint conversion | `references/checkpoint-splitting.md` |
| Validate a SeedOmni change | `references/validation.md` |

## Current Baseline

- Entry-time samples use `{"conversation_list": [...]}`. `seedomni_transform`
  runs ordered module CPU preprocessors (`OmniProcessor`) per sample in the
  Dataset stage, and `SeedOmniCollator` groups preprocessed samples into a
  micro-batch before GPU forward.
- `ModulePreprocessorBase` is a core data-flow contract, shared by training and
  inference. Training runs it inside `data_transform` (`seedomni_transform`);
  inference runs it once before the FSM.
- Modules communicate by mutating and returning the shared `conversation_list`
  carrier.
- Training graphs are flat edge lists over `module[.method]` endpoints.
  `to: end` is the virtual sink. Execution order is derived by topo sort.
- Generation graphs are FSMs. Each state body is an ordered inline edge list.
  `done` is framework-injected; use `module_signal` and `default` transitions.
- Store tokenizers, processors, and other assets in the owning module's
  subfolder.

## Templates

Templates under `templates/` are skeletons, not authoritative examples. Use them
to start a new config file, then compare against the live configs listed in
`references/module-contract.md`.

## Sibling Skills

- Use `veomni-new-model` first when adding a new HuggingFace model family under
  `veomni/models/transformers/`.
- Use this skill after that model exists, when wrapping it as SeedOmni
  modules or wiring it into Omni graphs.
- Use `veomni-develop` for unrelated framework work outside SeedOmni.
