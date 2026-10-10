# Adding a Model to SeedOmni

This page is the checklist for bringing a new model family into SeedOmni. Read
[Architecture](../design/architecture.md) first for the module split, the
conversation carrier and the graph views. Agents should use the
`/veomni-seed-omni` skill, which carries the same checklist plus review
references.

## 1. Split the checkpoint

Register a converter in `veomni/models/seed_omni/modules/<family>/convert_model.py`
(dispatched by `scripts/seed_omni/convert_model.py` on the upstream
`model_type`). It returns the upstream HF checkpoint as one module instance per
module, with processors and tokenizers attached through `attach_module_assets`,
plus the family's default graphs. `convert_checkpoint` saves the result through
`OmniModel.save_pretrained`: a root `config.json`, graph sidecars, and one
self-contained subfolder per module (`config.json`, `model.safetensors`, and any
processor or tokenizer files).

When every module is a key-prefix slice of the upstream checkpoint, register an
`OmniHFLayout` in `modules/<family>/hf_layout.py` under the upstream `model_type`
instead (`modules/qwen3/hf_layout.py` is the reference). Per module it declares
the `key_prefixes` map (source prefix → module prefix; the longest match wins),
`build_config` (HF config → module config) and `build_assets` (tokenizer /
processors); `tied_source_keys` names stored duplicates of tied weights. A
layout gives both the offline convert and the
[direct load and HF-layout export](training_and_inference.md#21-load-an-upstream-checkpoint-directly);
the contract is in [Upstream Checkpoint Layout](hf_layout.md).

## 2. Write each module

Under `veomni/models/seed_omni/modules/<family>/<sub>/`:

- `configuration.py`: a `PretrainedConfig` with a unique `model_type`.
- `modeling.py`: `class X(InferenceMixin, PretrainedOmniModule)`, pure
  HF-native. It holds `__init__`, the submodule layout and `forward`, and, if the
  module takes part in inference, an in-file `InferenceMixin` with the FSM
  `generate()` and its state. `InferenceMixin` must come **before**
  `PretrainedOmniModule` in the bases. The class must load and run under plain
  `from_pretrained` with no VeOmni import.
- `accelerated/accelerated.py`: `TrainingMixin` and `VeOmniMixin` (no
  `InferenceMixin`) plus IDE type stubs (below), and
  `class XAccelerated(VeOmniMixin, X)`. The Janus modules under
  `modules/janus/*/` are the reference pattern.
- `processing.py` (optional): when the module consumes raw images, audio or
  video.

Reuse the cross-family helpers in `modules/base/` where possible (text encoder,
packing, LLM generation).

### IDE type stubs

The `TrainingMixin` hooks call native modeling APIs through `self`, but the
mixin does not contain them; they live on the `modeling.py` class, which is only
mixed in at `XAccelerated(VeOmniMixin, X)`. Declare **only what the mixin's hooks
use**, so static analysis can follow the calls:

```python
class TrainingMixin(TrainingModuleMixin):
    config: BagelFlowConnectorConfig
    device: torch.device
    dtype: torch.dtype

    def embed_latent(
        self,
        latents: torch.Tensor,
        position_ids: torch.LongTensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """IDE stub — implemented on :class:`BagelFlowConnector` in ``modeling.py``."""
        ...
```

- Signatures match `modeling.py` exactly; the body is always `...`.
- Properties defined on `VeOmniMixin` in the same file use the docstring
  ``IDE stub — see :class:`VeOmniMixin` below (``config.field``).``
- Do not copy modeling logic into the mixin.
- `generate()` and its helpers live on the native class and need no stub unless a
  training hook calls them.

`modules/bagel/flow_connector/accelerated/accelerated.py` shows the full
convention.

## 3. Register

Register the classes in `veomni/models/seed_omni/modules/__init__.py`, keyed by
`model_type`:

| Registry | Holds |
|----------|-------|
| `OMNI_CONFIG_REGISTRY` | `configuration.py` classes |
| `OMNI_MODEL_REGISTRY` | native `modeling.py` classes |
| `OMNI_ACCELERATED_MODEL_REGISTRY` | `accelerated/accelerated.py` classes |
| `OMNI_PROCESSOR_REGISTRY` | `processing.py` classes |

The runtime resolves a module by reading its `config.json`, then `model_type`,
then the registry.

## 4. Write the YAML

Create `configs/seed_omni/<model>/<task>/` following
[the config layout](training_and_inference.md#1-config-layout): `base.yaml`,
`modules_train.yaml`, `graph_train.yaml`, plus shared `infer/` graphs and
`data.yaml`. Edges only declare order; conversation graphs move data through the
conversation list. Packed graphs (such as Janus `packed/graph_train.yaml`) call
`pack_*` methods and move packed tensors on the batch dict instead.

If the model needs a new data source, add a preprocessor as described in
[Data Format](data_format.md#custom-datasets).

## 5. Honour the contracts

- Return at most one scalar `_loss` per node, token-mean reduced over the
  module's own micro-batch.
- Read inputs in `pre_forward`, write results in `post_forward`. Conversation
  nodes return `{"conversation_list": ...}` so the carrier flows on.
- Implement `dummy_inputs()` for any encoder whose modality can be absent from a
  micro-batch, so FSDP stays aligned.
- Under Ulysses, slice in `pre_forward` and gather in `post_forward`; see
  [Sequence Parallelism](../distributed/sequence_parallel.md).
- For inference, emit `module_signal` strings to drive FSM transitions and clear
  private buffers in `reset_local_inference_state()` /
  `reset_global_inference_state()` / `finalize()`.
- Optional: mix in [Metric Meter](../mixins/metric_meter.md) for per-module
  FLOPs, and [Offline Encoding](../mixins/offline_encoding.md) for a cacheable
  encoder.

## 6. Validate

1. Render the graphs with `scripts/seed_omni/visualize_graph.py`
   ([Debugging the graph](training_and_inference.md#6-debugging-the-graph)).
2. Run the unit tests: `pytest tests/seed_omni/`.
3. Run a short train and inference end to end, then add a page under
   `docs/seed_omni/models/` and register it in `docs/index.md`.

Worked examples:

- [Janus](../models/janus.md): multimodal understanding and generation
  (SigLIP / VQVAE / LLaMA).
- [Qwen3](../models/qwen3.md): minimal text-only split, plus visual instruction
  tuning.
- [Qwen3 MoE](../models/qwen3_moe.md): MoE backbone with expert parallelism.
- [Qwen3-VL](../models/qwen3vl.md): vision-language split with per-module ViT.
- [BAGEL](../models/bagel.md): MoT backbone with flow-matching image generation.
