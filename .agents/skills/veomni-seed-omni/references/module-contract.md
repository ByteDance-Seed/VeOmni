# Module Contract, Stubs, and Workflow

Read this before adding or modifying `veomni/models/seed_omni/modules/**`. See
also `docs/seed_omni/usage/adding_a_model.md` and
`docs/seed_omni/design/architecture.md`.

## 1. Module Folder Shape

```text
modules/<family>/<submodule>/
├── configuration.py     # PretrainedConfig with unique model_type
├── modeling.py          # pure HF-native class + optional in-file InferenceMixin
├── accelerated/
│   ├── __init__.py      # re-exports the accelerated class for the registry
│   ├── accelerated.py   # TrainingMixin / MeterMixin / VeOmniMixin + XxxAccelerated
│   └── packed.py        # optional packed-graph hooks
└── processing.py        # optional ModulePreprocessorBase / processor wrapper
```

- **`modeling.py`**: pure HuggingFace-native, loadable with `from_pretrained` /
  `AutoModel`. If inference-capable, list `InferenceMixin` (or
  `SimpleArGenerationMixin` from `modules/base/llm_packing.py`) **before**
  `PretrainedOmniModule`: `class Xxx(InferenceMixin, PretrainedOmniModule)` so
  its `generate()` / `reset_*` / `finalize` methods take precedence over
  `PretrainedOmniModule`'s no-op defaults.
- **`accelerated/accelerated.py`**: defines `TrainingMixin(TrainingModuleMixin)`,
  optional `MeterMixin(MetricMeterMixin)`, `VeOmniMixin(BaseMixin,
  TrainingMixin, MeterMixin)`, and `class XxxAccelerated(VeOmniMixin, Xxx)`.
- **Registration** (`veomni/models/seed_omni/modules/__init__.py` at import
  time, keyed by `config.model_type`): `OMNI_CONFIG_REGISTRY`,
  `OMNI_MODEL_REGISTRY`, `OMNI_ACCELERATED_MODEL_REGISTRY`, and optional
  `OMNI_PROCESSOR_REGISTRY`.

## 2. Runtime Rules

- Graph endpoints and hooks return `dict[str, Any]`.
- Use `@pre_forward("<method>")` and `@post_forward("<method>")` when a module
  has multiple graph call-sites.
- Return loss under the exact key `"_loss"` (`OmniModel.forward()` pops
  `LOSS_KEY = "_loss"`), reduced to a scalar tensor over the module's
  micro-batch (decode losses reduce over `ps.fsdp_group`; see constraint 2 in
  `.agents/knowledge/veomni-seed-omni.md`).
- Mutate `ConversationItem` fields in place and return
  `{"conversation_list": conversation}` (or write `packed_*` tensors on the
  batch dict for packed graphs).
- Keep tokenizers and processors inside the owning module's subfolder via
  `get_assets()` / `attach_module_assets`.
- In training, every active node runs on every micro-batch: use CPU-preprocessor
  dummies or `dummy_inputs()` + a `0.0` anchor when a modality is absent.
- For shared stateless carrier helpers called by both `modeling.py` and
  `accelerated/accelerated.py`, define plain module-level functions in
  `modeling.py` (or reuse `modules/base/llm_packing.py`).

## 3. IDE Type Stubs on `TrainingMixin`

Because `VeOmniMixin` precedes `Xxx` in `XxxAccelerated(VeOmniMixin, Xxx)`'s
MRO, an unguarded `def foo(...): ...` on `TrainingMixin` shadows `Xxx.foo` at
runtime. Declare class-level attribute annotations (`config`, `device`, `dtype`,
modeling attributes) directly on `TrainingMixin`, and guard any typing-only
method or property stubs with `if TYPE_CHECKING:`:

```python
from typing import TYPE_CHECKING


class TrainingMixin(TrainingModuleMixin):
    config: XxxConfig
    device: torch.device
    dtype: torch.dtype

    if TYPE_CHECKING:

        def encode_helper(self, x: torch.Tensor) -> torch.Tensor:
            """IDE stub — implemented on :class:`Xxx` in ``modeling.py``."""
            ...
```

## 4. Optional Per-Module Metric Meter

Mix in `MetricMeterMixin` on `VeOmniMixin` (or `MeterMixin`):

- Call `self.metric_meter_set_seqlens("<method>", seqlens)` inside `pre_forward`
  **before** the `if sp_size > 1:` slice, passing the full pre-slice per-sample
  lengths (`OmniEnvironMeter` sums over `dp_group`, excluding SP; see constraint
  3 in `.agents/knowledge/veomni-seed-omni.md`).
- Implement `estimate_flops(seqlens)` counting only this module's compute.

## 5. Live Reference Map and Workflow

| Need | Path |
|---|---|
| In-tree minimal module A/B (native + accelerated) | `veomni/models/seed_omni/modules/fake_model/fake_module_a/`, `veomni/models/seed_omni/modules/fake_model/fake_module_b/` |
| In-tree minimal configs (train / eager & FSDP infer / graphs) | `configs/seed_omni/fake_model/train/`, `configs/seed_omni/fake_model/infer/` |
| Family modules (vision, text_encoder, AR backbone, VQ/VAE) | `veomni/models/seed_omni/modules/<family>/<submodule>/` |
| Family configs (train, packed, infer) | `configs/seed_omni/<Family>/<variant>/` |

Workflow:
1. Pick the closest live module and decide asset ownership, item tags (`type` /
   `role` / `meta`), dummy injection, and graph endpoints.
2. Implement `configuration.py`, `modeling.py`, `accelerated/accelerated.py`,
   and optional `processing.py`, then register in `modules/__init__.py`.
3. Wire `modules_train.yaml`, `graph_train.yaml`, and `infer/` YAML configs.
4. Validate with `pytest tests/seed_omni/ -q` and `make quality`.
