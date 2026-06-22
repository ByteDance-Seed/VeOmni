# SeedOmni V2 Parity Suite

`parity_suite` is the shared pytest harness for checking SeedOmni V2 behavior against an independent reference implementation. Model-specific tests own their pytest entrypoints, YAML contracts, and small `ParityDriver` implementations. This package provides reusable discovery, reference capture, V2 tier runners, launcher integration, and comparison helpers; it is not itself a model parity entrypoint.

## Directory Layout

```text
tests/seed_omni/parity_suite/
├── pytest_entrypoint.py      Factory for model-owned pytest parity entrypoints
├── runner.py                 End-to-end execution for one parity case
├── driver/                   Base class package for model-specific parity drivers
│   ├── base.py               ParityDriver composition and dtype/determinism
│   ├── reference.py          Reference loading and recipe execution
│   ├── v2_loading.py         V2 model/module loading
│   ├── requests.py           V2 request dispatch via build_{kind}_request hooks
│   ├── runtime.py            Optional driver runtime hooks
│   └── observations.py       Train observation and gradient helpers
├── core/                     Shared parity helpers
│   ├── config/               YAML spec loading, discovery, gating, probes
│   ├── compare.py            Metrics, comparison, and reporting
│   ├── runtime.py            Determinism, device moves, and training helpers
│   └── fixtures.py           Model-agnostic deterministic fixtures
├── reference/                Reference model loading, hook capture, tensor normalization, output contract
├── v2/                       Shared V2 helpers
│   ├── model.py              V2 config/module loading
│   ├── observation.py        Observer sinks and V2-side capture hooks
│   ├── tier_runners/         Graph, module, and framework tier runners
│   │   └── framework_support.py
│   │                           Local trainer/report helpers used by framework wrappers
│   └── workers/              Framework subprocess runners and torchrun worker entrypoints
├── launcher/                 Optional subprocess GPU-pool launcher used from pytest
└── suite_tests/              Unit tests for the harness itself
```

Model-owned parity inputs live beside the model tests, not under this package. For example, BAGEL uses:

```text
tests/seed_omni/bagel/
├── contracts/                BAGEL-local contract and helper tests
└── parity/
    ├── test_parity_cases.py  BAGEL-owned pytest entrypoint
    ├── base.yaml             Reference and V2 model paths, enabled tiers, tolerances, launcher config
    ├── probes.yaml           Public probe names mapped to V2 node/field and reference observations
    ├── recipes/*.yaml        Stimuli, tier runs, selected probes, run options, gates, V2 load overrides
    ├── reference/data/       BAGEL reference input, generation-option, and train-batch adapters
    └── driver.py             BAGEL-specific reference and V2 adaptation
```

## Data Contract Boundaries

Shared `parity_suite` code is model-agnostic. It owns discovery, gating, reference oracle execution, tier dispatch, probe resolution, tolerance comparison, reports, and abstract adapter method names. It must not require model-internal tensor layouts in recipes, probe paths, runner logic, or base driver defaults.

Prefer importing helpers from their owning module (`v2.observation`, `v2.model`, `reference.oracles.hf_model`, and so on) instead of high-level package re-exports. Top-level packages are intentionally thin namespaces so internal test-runtime helpers do not become accidental public APIs.

Model drivers own concrete input adaptation. A driver may start from raw fixtures, model-owned canonical data, or reference outputs, but shared suite code treats those values as opaque. Standard conversation-carrier recipes can declare ``stimulus.conversation_list`` for a single sample or ``stimulus.batched_conversation_list`` for an explicit batch and use the default request builder. For V2 inference requests, ``stimulus.conversation_list`` materializes to the production ``{"conversation_list": list[ConversationItem]}`` shape; for V2 training requests, it materializes to ``{"conversation_list": list[list[ConversationItem]]}``. Implement ``build_{reference.kind}_request`` hook methods on ``ParityDriver`` subclasses only when a case needs non-standard request adaptation. Hook methods should return common V2 request keys, usually ``{"conversation_list": ...}``, and leave model-internal conversion to model runtime hooks or model-specific helpers.

Reference oracles return the suite contract defined in ``reference/oracles/contract.py``:

```python
ReferenceRunResult(canonical={...}, observations={...})
```

- ``canonical`` is the model-owned payload passed to V2 request handlers.
- ``observations`` holds values addressed by ``probes.yaml`` reference fields such as ``ref.field: hidden_state``.
- ``None`` remains valid only when reference oracle execution is skipped and request handlers build from recipe stimulus.
- ``ParityReport`` remains valid for reference-only recipe execution and is not routed through this contract.

Future models such as Janus should configure ``reference.hf_model`` or ``reference.hf_module`` backends, then define ``build_{kind}_request`` hook methods keyed by ``reference.kind`` only when default conversation-list dispatch is insufficient.

## Reference Capture

Reference capture has one outer orchestration pass and three observation sources:

- Field taps read named values returned by the reference subject in `ReferenceRunResult.observations`.
- Hook taps are suite-owned forward hooks declared by `probes.yaml`; `capture_hook_taps(...)` installs them on configured reference submodules.
- Observation adapters are subject-owned helpers such as `MethodPatchObservationAdapter`; a reference subject installs them around its official methods when a value is easier to record inside the model's own inference or training path.

The outer `ReferenceOracle.capture(...)` owns lifecycle: load the reference subject, run the official path with the selected taps/adapters, materialize small observed values to CPU, release the reference subject, then verify memory was freed before V2 execution. Observation adapters should only record evidence; they should not reimplement model runtime semantics.

## Tier Boundaries

Module tier is the inference-forward module-boundary tier. It validates one module, or a very small module-local call chain, by comparing forward observations produced by the independent reference inference path against V2 module outputs and carrier mutation. Module-tier recipes should use only minimal model-owned micro graphs under the model's test config directory; they must not point at full-model production graph configs as the meaning of the tier. The V2 side loads only modules referenced by the selected graph. This keeps module-tier runs lazy and scoped: a text-encoder micro graph should not force SigLIP, VAE, MoT, or flow modules to be constructed unless the micro graph actually references them. Module tier does not own training loss parity, backward parity, optimizer behavior, or distributed safety.

Reference execution for module tier should call the model's highest-level official inference entrypoint that naturally reaches the boundary under test. For example, BAGEL module references should call `InterleaveInferencer.interleave_inference(...)` and install non-mutating hooks around official calls, rather than manually replaying `init_gen_context`/`update_context_*`/`gen_*` loops or reimplementing packed layout, attention, VAE, flow, or loss semantics in the oracle. The shared `hf_module.<name>` reference backend is a facade over `reference.hf_model`: by default it loads the full HF subject for compatibility, but model subjects can override `create_hf_module_subject()` or `HfModuleSubject.load_reference_subject()` to lazily assemble only the backing pieces needed for the requested module boundary. The module facade is released before V2 loading so reference memory does not remain live across the comparison.

Graph tier is the eager graph-oracle tier. For training, a graph-tier parity case is meaningful when both the independent reference and the V2 `OmniModel.forward()` graph can run as an eager reference path and compare the selected loss and gradient probes. Graph tier validates graph dispatch, carrier mutation, loss aggregation, and graph-level backward semantics against the reference. It is not a scalability guarantee for models whose reference or V2 graph cannot fit in eager form.

Framework tier is the trainer and distributed-training tier. It validates `OmniTrainer`, optimizer and scheduler updates, clipping, checkpointing, data health, and FSDP/HSDP behavior. When an eager graph oracle is runnable, framework policies may compare trainer/FSDP results to that direct graph baseline. For models that are too large for eager graph/reference execution, training parity should be defined directly at the framework/distributed level against a sharded or otherwise low-memory reference path rather than forcing graph tier to become an FSDP test.

Bare model-internal packer checks and train-only module loss checks should live in model-specific unit tests, graph-tier cases, or framework-tier cases outside module-tier reference flow.

## Running Suite Tests

`parity_suite` does not own model parity entrypoints. Its own pytest surface is the harness unit test
suite:

```bash
source .venv/bin/activate
python -m pytest -q tests/seed_omni/parity_suite
```

## Running Model Cases

Run model parity through the model-owned entrypoint. For example, BAGEL runs from the Open-VeOmni
repository root with:

```bash
source .venv/bin/activate
export VEOMNI_V2_TEST_ENABLE_PARITY_CHECK=1

python -m pytest -q tests/seed_omni/bagel/parity/test_parity_cases.py
```

To run one discovered case, pass the parametrized pytest id:

```bash
python -m pytest -q \
  'tests/seed_omni/bagel/parity/test_parity_cases.py::test_bagel_parity_case[bagel.text_und.graph.one_step]'
```

The default gate skips full parity execution unless `VEOMNI_V2_TEST_ENABLE_PARITY_CHECK=1` is set. Case gates can also require CUDA, a minimum CUDA device count (`gate.min_cuda_devices`, used by multi-rank FSDP2/HSDP cases), a reference checkpoint, or a V2 model path.

## GPU Launcher

If a model sets `launcher.enable_parallel: true` in `base.yaml`, pytest groups the selected runnable cases and executes them in subprocesses with a simple CUDA device pool. Launcher logs are written under:

```text
outputs/parity_suite/launcher/
```

Directly selecting a single parametrized case bypasses the grouped launcher so the case can be debugged in the foreground.

## Adding Or Updating A Case

1. Add or update the model test contract under `tests/seed_omni/<model>/parity/`.
   The model parity package should define its own pytest entrypoint with the shared factory:

   ```python
   from pathlib import Path

   from tests.seed_omni.parity_suite.pytest_entrypoint import make_parity_test


   test_<model>_parity_case = make_parity_test(Path(__file__).parent)
   ```

2. Configure `base.yaml` with reference oracle backends, V2 model targets, enabled graph names, enabled tiers, tolerances, gates, and launcher settings.
   Reference backends and V2 configs are target registries. Recipe variants select one target with `reference.oracle`; the suite uses the same target key to pick the V2 config:

   ```yaml
   reference:
     hf_model:
       module: tests.seed_omni.foo.parity.reference.hf_model:FooReferenceSubject
       checkpoint: /path/to/foo/reference
     hf_module:
       - text_encoder

   v2_model:
     model_root: /path/to/foo/seed_omni
     hf_model:
       config_dir: configs/seed_omni/Foo/full_model
     hf_module:
       text_encoder:
         config_dir: tests/seed_omni/foo/parity/configs/text_encoder
   ```

   `reference.hf_module` entries are named facades backed by `reference.hf_model`; they do not declare separate checkpoints. Use them when a recipe should compare a module-boundary view while still executing the official reference path.

3. Add recipe variants in `recipes/*.yaml`. Each variant declares exactly one of `stimulus` or `data`, then lists `runs` by tier. The `probes` list names public probe keys from `probes.yaml`. `stimulus` is driver-owned and opaque to the shared suite.
   Recipe variants may declare only the narrow `v2_model.module_overrides` block. Use it for parity-only module loading differences, such as keeping a VAE on CPU/fp32 to match an official app path, while the rest of the V2 modules use the runner device and dtype. Overrides may name only modules referenced by the selected graph. Put target-specific model roots and config directories under `base.yaml`, then select them with `reference.oracle: hf_module.<name>` or `reference.oracle: hf_model`.

   ```yaml
   v2_model:
     module_overrides:
       bagel_vae:
         device: cpu
         dtype: float32
   ```

   Module-level cases should use the shared module runner only with a module-local inference micro graph that reaches the requested observation boundary. Do not use module tier as a shorthand for a full production inference graph or for a train/loss graph.
   The shared observation path records whitelisted tensor fields from node outputs and from returned `conversation_list` carriers: `value` reads `item.value`, while meta fields such as `input_ids`, `labels`, and `attention_mask` read `item.meta[...]`.
   Drivers should add extra observations only for model-specific derived values.
   Reference-side module observations should be collected by calling official high-level inference APIs and installing capture-only hooks. A reference adapter may translate suite stimuli into official inputs and rename captured tensors into probe fields, but it must not manually implement the model's inference loop, packing rules, forward math, loss formulas, or sampling semantics.
   If a future graph/framework path needs carrier observations, extend the shared observation helpers first instead of adding per-node carrier readers in a model driver.
   The shared V2 loader builds a graph-active config and loads only the modules used by that config. With a `model_root`, each active module is loaded from `<model_root>/<module_name>`. If the selected V2 target has no `model_root`, the loader can instantiate active modules from node-specific `modules_train.yaml` `parity` blocks containing `model_type`, `config`, and an optional `setup` callable.
   Standard conversation-list module cases should declare `stimulus.conversation_list` as a single sample:

   ```yaml
   stimulus:
     conversation_list:
       - type: image
         role: user
         value:
           kind: random
           shape: [3, 4, 4]
           distribution: uniform
           seed: 11
           dtype: float
       - type: text
         role: user
         value: "Describe this image."
       - type: output
         role: assistant
         value:
           kind: tensor
           tensor: [1, 10, 2]
           dtype: long
   ```

   Use `stimulus.batched_conversation_list` only when the case intentionally validates `bs > 1`; it is authored as `list[list[item spec]]`. The suite normalizes single-sample `stimulus.conversation_list` to a batched shape for reference oracle execution, to `list[ConversationItem]` for V2 inference, and to `list[list[ConversationItem]]` for V2 training.

   Each item spec may include `type`, `role`, `value`, `source`, and `meta`. Supported item `type` values are `image`, `text`, and `output`. Text items use the production carrier shape directly: `value` is a plain string. Non-text values that YAML cannot express directly still use the suite materializer tagged union: `kind: tensor` requires `tensor` and optional `dtype`/`device`; `kind: random` requires `shape` and supports `distribution: uniform|normal|zeros|ones`, `seed`, `dtype`, and distribution parameters; `kind: linspace` requires `start` and `end`; `kind: image` creates a deterministic PIL image. Legacy untagged values such as `{tensor: [...]}` are rejected. Prefer deterministic `kind: random` image specs over large inline image tensors; keep explicit tensors for semantic token ids or small training targets where readability matters.

Each run can be toggled independently with `enable`. The field is optional and defaults to `true`; when set to `false`, discovery filters the run before pytest parametrization, so it will not appear in `--collect-only` output. Use a YAML bool, not a quoted string:

```yaml
interleave_image_gen:
  - stimulus:
      prompt: "A glass greenhouse filled with tiny orange trees."
    reference:
      oracle: hf_model
      kind: image_gen
    runs:
      graph:
        - id: image_span_one_step
          enable: true
          probes:
            - image.velocity
      module:
        - id: image_span_one_step
          enable: false
          probes:
            - image.velocity
```

4. Add probes in `probes.yaml`. Each top-level key is a public probe name. The probe maps a V2 graph node and observation field to a reference observation field and tolerance policy:

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
    field: hidden_state              # reference oracle observations["hidden_state"]
    hook: model.norm                 # hf_model-only module hook path
    extractor: pkg.mod:fn            # hf_model-only callable entrypoint
  tol: hidden                        # required tolerance policy key from base.yaml
  step: last                         # optional; `last` (default) or `all`
```

Declare exactly one of `ref.field`, `ref.hook`, or `ref.extractor`.
5. Implement or extend `driver.py` by returning a `ParityDriver` from `create_driver(case)`. Reference execution is selected by `recipe.reference.oracle` and routed through the configured `reference.hf_model` or `reference.hf_module.<name>` backend; drivers only override `reference_oracle()` for genuinely model-specific runtimes.
   If a model file binds `sdpa_kernel` as a module global, override `runtime_sdpa_kernel_modules()` and return every reference/V2 Python module whose global should follow `run.options.deterministic_sdpa`. This is one runtime policy hook, not separate reference and V2 behavior.
6. Run the harness unit tests and at least one targeted parity case before broadening to the full suite.

Useful checks while editing the harness:

```bash
python -m pytest -q tests/seed_omni/parity_suite/suite_tests
python -m pytest -q tests/seed_omni/<model>/parity/test_parity_cases.py --collect-only
```

## Execution Model

Each discovered `ParityCase` combines one model spec, one recipe variant, one run, one graph, and the graph node catalog. `runner.run_parity_case()` loads the model-specific driver, captures reference oracle observations when the effective gate requires them, dispatches the V2 tier, then compares each selected probe with the tolerance policy from `probes.yaml` and `base.yaml`.

Reference-only recipes can return their own `ParityReport`. Graph, module, and framework tiers usually return V2 observations keyed by `(state, node)` so the shared runner can compare them against captured reference observations.

Graph-tier call chain:

```text
pytest case
  -> runner.run_parity_case(case)
  -> driver.reference_oracle().capture(...)
     -> hf_model / hf_module subject executes the official reference path
     -> reference capture materializes field, hook, and extractor taps
  -> runner._run_v2(...)
  -> v2.tier_runners.graph.run_v2_*_graph(...)
  -> driver.v2_request_kwargs(...)
     -> default conversation request when stimulus.conversation_list is present
     -> build_{reference.kind}_request only for non-standard requests
  -> OmniModel.generate(...) or OmniModel.forward(...)
  -> V2 observation capture records whitelisted node outputs/carrier fields
  -> runner compares mapped probes
```

Module-tier call chain:

```text
pytest case
  -> runner.run_parity_case(case)
  -> hf_module.<name> reference oracle
     -> loads the configured hf_model subject or a model-specific lazy facade
     -> executes the official high-level reference path and filters module fields
  -> v2.tier_runners.module.run_v2_infer_module(...)
  -> load the graph-active V2 modules for the selected module micro graph
  -> driver.v2_request_kwargs(...)
  -> run_generation_fsm(...) steps the production SeedOmni generation graph
  -> ModuleRuntime records only required node/state observations
  -> runner compares mapped probes
```
