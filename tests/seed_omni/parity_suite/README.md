# SeedOmni V2 Parity Suite

`parity_suite` is the shared pytest harness for checking SeedOmni V2 behavior against an independent reference implementation. Model-specific tests provide YAML contracts and a small `ParityDriver`. This package discovers those contracts, captures reference probes, runs the selected V2 graph/module/framework tier, and compares the observed values.

## Directory Layout

```text
tests/seed_omni/parity_suite/
├── test_parity_cases.py      Pytest collection entrypoint for discovered cases
├── runner.py                 End-to-end execution for one parity case
├── driver.py                 Base class for model-specific parity drivers
├── core/                     YAML spec loading, discovery, gating, mapping, metrics, reports
├── reference/                Reference model loading, hook capture, tensor normalization
├── v2/                       Shared V2 helpers
│   ├── model.py              V2 config/module loading
│   ├── observation.py        Observer sinks and capture hooks
│   ├── tier_runners/         Graph, module, and framework tier runners
│   └── workers/              Torchrun subprocess entrypoints for framework tier
├── launcher/                 Optional subprocess GPU-pool launcher used from pytest
└── suite_tests/              Unit tests for the harness itself
```

Model-owned parity inputs live beside the model tests, not under this package. For example, BAGEL uses:

```text
tests/seed_omni/bagel/
├── base.yaml                 Reference and V2 model paths, enabled tiers, tolerances, launcher config
├── probes.yaml              Probe names mapped to V2 observation fields and reference taps
├── recipes/*.yaml            Stimuli, tier runs, selected probes, run options, per-run gates
└── driver.py                 BAGEL-specific reference and V2 adaptation
```

## Data Contract Boundaries

Shared `parity_suite` code is model-agnostic. It owns discovery, gating, reference capture, tier dispatch, probe mapping, tolerance comparison, reports, and abstract adapter method names. It must not require model-internal tensor layouts in recipes, mapping paths, runner logic, or base driver defaults.

Model drivers own concrete input adaptation. A driver may start from raw fixtures, model-owned canonical data, or reference outputs, but shared suite code treats those values as opaque. Driver adapters such as `v2_infer_request()` and `v2_train_batch_kwargs()` should return common V2 request keys, usually `{"conversation_list": ...}`, and leave model-internal conversion to model runtime hooks or model-specific helpers.

For training parity, graph and current module-tier runs execute through SeedOmni graph nodes and invoke `pre_forward` / `post_forward` hooks. The current module tier is node-level parity, not a bare packed-tensor module API test. Bare model-internal packer checks should live in model-specific unit tests outside shared suite flow.

Framework-tier training checks are V2 runtime policy checks by default. They validate trainer behavior, optimizer and scheduler updates, checkpointing, distributed/FSDP execution, and data health. They should not compare to an official oracle unless a case explicitly keeps reference capture enabled and declares probe mappings for that purpose.

## Running Cases

Run from the Open-VeOmni repository root:

```bash
source .venv/bin/activate
export VEOMNI_V2_TEST_ENABLE_PARITY_CHECK=1

python -m pytest -q tests/seed_omni/parity_suite/test_parity_cases.py
```

To run one discovered case, pass the parametrized pytest id:

```bash
python -m pytest -q \
  'tests/seed_omni/parity_suite/test_parity_cases.py::test_seed_omni_v2_parity_case[bagel.text_und.graph.one_step]'
```

The default gate skips full parity execution unless `VEOMNI_V2_TEST_ENABLE_PARITY_CHECK=1` is set. Case gates can also require CUDA, a reference checkpoint, or a V2 model path.

## GPU Launcher

If a model sets `launcher.enable_parallel: true` in `base.yaml`, pytest groups the selected runnable cases and executes them in subprocesses with a simple CUDA device pool. Launcher logs are written under:

```text
outputs/parity_suite/launcher/
```

Directly selecting a single parametrized case bypasses the grouped launcher so the case can be debugged in the foreground.

## Adding Or Updating A Case

1. Add or update the model test contract under `tests/seed_omni/<model>/`.
2. Configure `base.yaml` with the reference loader, V2 model config, enabled graph names, enabled tiers, tolerances, gates, and launcher settings.
3. Add recipe variants in `recipes/*.yaml`. Each variant declares exactly one of `stimulus` or `data`, then lists `runs` by tier. The `probes` list names entries from `probes.yaml`.
4. Add probe mappings in `probes.yaml`. Each mapping selects the graph node, V2 observation field, reference tap, tolerance policy, and optional state or step policy.
5. Implement or extend `driver.py` by returning a `ParityDriver` from `create_driver(case)`. The driver owns reference loading, reference execution, V2 model loading, and any model-specific input/output adaptation.
6. Run the harness unit tests and at least one targeted parity case before broadening to the full suite.

Useful checks while editing the harness:

```bash
python -m pytest -q tests/seed_omni/parity_suite/suite_tests
python -m pytest -q tests/seed_omni/parity_suite/test_parity_cases.py --collect-only
```

## Execution Model

Each discovered `ParityCase` combines one model spec, one recipe variant, one run, one graph, and the graph node catalog. `runner.run_parity_case()` loads the model-specific driver, captures reference taps when the effective gate requires them, dispatches the V2 tier, then compares each selected probe with the tolerance policy from `probes.yaml` and `base.yaml`.

Reference-only recipes can return their own `ParityReport`. Graph, module, and framework tiers usually return V2 observations keyed by `(state, node)` so the shared runner can compare them against captured reference taps.
