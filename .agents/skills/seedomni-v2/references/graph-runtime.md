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

`base.yaml` points to module and graph files. The graph loader accepts either a
wrapped payload (`training_graph:` / `generation_graph:`) or a bare payload at
the file top level. Preserve the surrounding style when editing an existing
file, and do not redeclare unrelated launcher config in graph files.

## Training Graph

Training graph YAML is a flat edge list:

```yaml
training_graph:
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

Generation graph YAML is an FSM. The payload fields are `initial` and `states`:

- `generation_graph.states.<state>.body` is an ordered inline edge list.
- In bare files, use top-level `initial:` and `states:`.
- In wrapped files, put those under `generation_graph:`.
- State transitions use `module_signal` or `default`.
- `default` must be the last transition.
- Do not declare a `done` state. It is framework-injected.
- Use module code to emit semantic signals such as `text_done` or
  `image_complete`; do not make the FSM inspect raw token IDs.

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
