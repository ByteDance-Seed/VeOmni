# SeedOmni V2 Parity Suite

`parity_suite` is the shared pytest harness for checking SeedOmni V2 behavior against an independent reference implementation. Model-specific tests provide YAML contracts and a small `ParityDriver`. This package discovers those contracts, captures reference probes, runs the selected V2 graph/module/framework tier, and compares the observed values.

## Directory Layout

```text
tests/seed_omni/parity_suite/
├── test_parity_cases.py      Pytest collection entrypoint for discovered cases
├── runner.py                 End-to-end execution for one parity case
├── driver/                   Base class package for model-specific parity drivers
│   ├── base.py               ParityDriver composition and dtype/determinism
│   ├── reference.py          Reference loading and recipe execution
│   ├── v2_loading.py         V2 model/module loading
│   ├── requests.py           V2 request dispatch via build_{kind}_request hooks
│   ├── runners.py            Graph/module/framework tier dispatch wrappers
│   └── observations.py       Train observation and gradient helpers
├── core/                     Shared parity helpers
│   ├── config/               YAML spec loading, discovery, gating, probes
│   ├── compare.py            Metrics, comparison, and reporting
│   ├── runtime.py            Determinism, device moves, and training helpers
│   └── fixtures.py           Model-agnostic deterministic fixtures
├── reference/                Reference model loading, hook capture, tensor normalization, output contract
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
├── probes.yaml              Public probe names mapped to V2 node/field and reference taps
├── recipes/*.yaml            Stimuli, tier runs, selected probes, run options, per-run gates
└── driver.py                 BAGEL-specific reference and V2 adaptation
```

## Data Contract Boundaries

Shared `parity_suite` code is model-agnostic. It owns discovery, gating, reference capture, tier dispatch, probe resolution, tolerance comparison, reports, and abstract adapter method names. It must not require model-internal tensor layouts in recipes, probe paths, runner logic, or base driver defaults.

Model drivers own concrete input adaptation. A driver may start from raw fixtures, model-owned canonical data, or reference outputs, but shared suite code treats those values as opaque. Implement ``build_{reference.kind}_request`` hook methods on ``ParityDriver`` subclasses and resolve them through ``v2_request_kwargs()``. Hook methods should return common V2 request keys, usually ``{"conversation_list": ...}``, and leave model-internal conversion to model runtime hooks or model-specific helpers.

Normal reference handlers must return the suite contract defined in ``reference/contract.py``:

```python
{"canonical": {...}, "reference": {...}}
```

- ``canonical`` is the model-owned payload passed to V2 request handlers through ``canonical_from_reference_output()``.
- ``reference`` holds values addressed by ``probes.yaml`` reference fields such as ``ref.field: hidden_state`` (resolved to ``reference.hidden_state``).
- ``None`` remains valid only when reference capture is skipped and request handlers build from recipe stimulus.
- ``ParityReport`` remains valid for reference-only recipe execution and is not routed through this contract.

Future models such as Janus should implement reference loading plus reference handlers that return this shape, then define ``build_{kind}_request`` hook methods keyed by ``reference.kind``.

## Tier Boundaries

Module tier is the local module-behavior tier. It may use lazy module loading and CPU offload so only the active module is resident on the target device while the carrier stays materialized between nodes. Inference module-tier runs validate per-node FSM behavior. Training module-tier runs validate forward/loss behavior through module `pre_forward` / call / `post_forward` hooks, but they do not own training gradient parity.

Graph tier is the eager graph-oracle tier. For training, a graph-tier parity case is meaningful when both the independent reference and the V2 `OmniModel.forward()` graph can run as an eager reference path and compare the selected loss and gradient probes. Graph tier validates graph dispatch, carrier mutation, loss aggregation, and graph-level backward semantics against the reference. It is not a scalability guarantee for models whose reference or V2 graph cannot fit in eager form.

Framework tier is the trainer and distributed-training tier. It validates `OmniTrainer`, optimizer and scheduler updates, clipping, checkpointing, data health, and FSDP/HSDP behavior. When an eager graph oracle is runnable, framework policies may compare trainer/FSDP results to that direct graph baseline. For models that are too large for eager graph/reference execution, training parity should be defined directly at the framework/distributed level against a sharded or otherwise low-memory reference path rather than forcing graph tier to become an FSDP test.

Bare model-internal packer checks should live in model-specific unit tests outside shared suite flow.

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

The default gate skips full parity execution unless `VEOMNI_V2_TEST_ENABLE_PARITY_CHECK=1` is set. Case gates can also require CUDA, a minimum CUDA device count (`gate.min_cuda_devices`, used by multi-rank FSDP2/HSDP cases), a reference checkpoint, or a V2 model path.

## GPU Launcher

If a model sets `launcher.enable_parallel: true` in `base.yaml`, pytest groups the selected runnable cases and executes them in subprocesses with a simple CUDA device pool. Launcher logs are written under:

```text
outputs/parity_suite/launcher/
```

Directly selecting a single parametrized case bypasses the grouped launcher so the case can be debugged in the foreground.

## Adding Or Updating A Case

1. Add or update the model test contract under `tests/seed_omni/<model>/`.
2. Configure `base.yaml` with the reference loader, V2 model config, enabled graph names, enabled tiers, tolerances, gates, and launcher settings.
3. Add recipe variants in `recipes/*.yaml`. Each variant declares exactly one of `stimulus` or `data`, then lists `runs` by tier. The `probes` list names public probe keys from `probes.yaml`. `stimulus` is driver-owned and opaque to the shared suite.
4. Add probes in `probes.yaml`. Each top-level key is a public probe name. The probe maps a V2 graph node and observation field to a reference tap and tolerance policy:

```yaml
text.hidden:
  v2:
    node: bagel_qwen2_mot.generate   # required V2 graph node / call-site
    state: prompt_encode             # optional FSM-state disambiguation
    field: bagel_last_hidden_state   # required V2 observation field (`loss` aliases to the shared loss field)
    grad:                            # required for gradient probes
      parameter: lm_head.weight      # required gradient parameter path
      module: bagel_text_encoder     # optional; defaults to the node prefix before the first dot
  ref:
    field: hidden_state              # reference output path `reference.<field>`
    hook: model.norm                 # or a module hook path
    extractor: pkg.mod:fn            # or a callable entrypoint
  tol: hidden                        # required tolerance policy key from base.yaml
  step: last                         # optional; `last` (default) or `all`
```

Declare exactly one of `ref.field`, `ref.hook`, or `ref.extractor`.
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
