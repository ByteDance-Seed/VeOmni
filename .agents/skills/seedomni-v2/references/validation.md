# Validation

Use the narrowest validation that proves the changed surface, then broaden.

## Static And Graph Checks

```bash
source .venv/bin/activate
python scripts/visualize_omni_graph.py configs/seed_omni/<model>/<variant>/base.yaml
python scripts/visualize_omni_graph.py configs/seed_omni/<model>/<variant>/base.yaml --visualize.format html
```

Check:

- Training sources are expected carrier/data entry nodes.
- All sinks flow to `end`.
- No training cycles exist.
- Generation states use valid `module_signal` or `default` transitions.
- `default` transitions are last.

## Focused Runtime Checks

- Import the module and config directly.
- Build the target `OmniConfig`.
- Run the smallest available unit or smoke for the module.
- For graph edits, run the graph visualization before launching GPU work.

## Broader Checks

```bash
pytest tests/seed_omni/ -q
make quality
```

For CUDA/GPU-dependent commands, use sandbox escalation in this workspace so
device nodes and NVML are available.

## Janus Smoke Baseline

When the task affects shared graph/module behavior, compare against Janus smoke
scripts if practical:

```bash
bash test_und.sh
bash test_gen.sh
bash test_train.sh
```

Keep outputs under `outputs/`.
