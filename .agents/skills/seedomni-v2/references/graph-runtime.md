# Graph Runtime And YAML

Read this before editing `configs/seed_omni/**` graph or module files.

## File Split

Typical layout:

```text
configs/seed_omni/<Model>/<variant>/
├── base.yaml
├── modules_train.yaml
├── modules_infer_eager.yaml
├── modules_infer_fsdp.yaml
├── graph_train.yaml
├── graph_infer.yaml
└── graph_infer_<scenario>.yaml
```

`base.yaml` points to module and graph files. **A graph file *is* its graph** —
the payload sits at the file top level with no wrapper key. Do not redeclare
unrelated launcher config in graph files.

## Training Graph

Training graph YAML is a flat edge list at the top level:

```yaml
- { from: vision_encoder, to: backbone }
- { from: text_encoder.encode, to: backbone }
- { from: backbone, to: text_encoder.decode }
- { from: text_encoder.decode, to: end }
```

Rules:

- Endpoints are `module[.method]` strings.
- Bare module names use the framework default method.
- Active nodes are derived from edge endpoints.
- Every sink needs an outgoing edge to `end`.
- Topological sort determines execution order.
- Edges declare dependency order. Data flows through the shared carrier.

## Generation Graph

Generation graph YAML is an FSM, with `initial:` and `states:` at the file top
level:

- `states.<state>.body` is an ordered inline edge list.
- State transitions use `module_signal` or `default`.
- `default` must be the last transition.
- Do not declare a `done` state. It is framework-injected.
- Use module code to emit semantic signals such as `text_done` or
  `image_complete`; do not make the FSM inspect raw token IDs.

One authoring file declares one scenario. `infer.infer_graph` maps every scenario
name to its file, and `infer.infer_type` picks the active one:

```yaml
infer:
  infer_graph:
    infer_gen: configs/seed_omni/<Model>/<variant>/graph_infer_gen.yaml
    infer_und: configs/seed_omni/<Model>/<variant>/graph_infer_und.yaml
  infer_type: infer_gen
```

## Generation Graphs In A Checkpoint

`OmniConfig` stores **every** scenario, not just the active one, so an exported
checkpoint is not locked to the scenario it was exported under:

- `config.generation_graphs` — `{infer_type: fsm_spec}`, all scenarios.
- `config.infer_type` — the active scenario (unset means the first declared).
- `config.generation_graph` — read-only property returning the active FSM. This
  is what `OmniModel` binds; switching scenario means setting `infer_type` and
  rebuilding the model.

The checkpoint sidecar `generation_graph.yaml` therefore wraps a **map**:

```yaml
generation_graphs:
  infer_gen: {initial: ..., states: {...}}
  infer_und: {initial: ..., states: {...}}
```

The older single `generation_graph:` sidecar layout is rejected at load time —
re-export with `scripts/seed_omni/export_omni_checkpoint.py`. The wrapper exists
only in the checkpoint sidecar, which needs a key to hold a *map* of scenarios;
a per-scenario **authoring** file is the bare FSM.

## Module Config

Module files map module names to model paths and optional per-module training or
accelerator settings. `model_type` belongs in the module checkpoint
`config.json`, not in YAML.

## Templates

Use:

- `templates/base.template.yaml`
- `templates/modules_train.template.yaml`
- `templates/modules_infer_eager.template.yaml`
- `templates/modules_infer_fsdp.template.yaml`
- `templates/graph_train.template.yaml`
- `templates/graph_infer.template.yaml`

After copying a template, compare against the live Janus config for exact field
names used on the current branch.
