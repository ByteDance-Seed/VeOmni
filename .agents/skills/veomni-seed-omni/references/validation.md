# Validation

Use the narrowest validation that proves the changed surface, then broaden.

## Static And Graph Checks

Inspect the resolved `TrainingGraph` and `GenerationGraph` for a launcher YAML:

- Training sources are expected carrier/data entry nodes.
- All sinks flow to `end`.
- No training cycles exist.
- Generation states use valid `module_signal` or `default` transitions.
- `default` transitions are last.

## Focused Runtime Checks

- Import the module and config directly.
- Build the target `OmniConfig`.
- Run the smallest available unit or smoke test for the module.

## Broader Checks

```bash
pytest tests/seed_omni/ -q
make quality
```
