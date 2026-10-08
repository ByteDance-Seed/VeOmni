# Offline Encoding (`OfflineEncodingMixin`)

`OfflineEncodingMixin` lets a SeedOmni module run an expensive, deterministic
encoder, such as a VAE or a ViT, once offline, and train from the cached tensors
afterwards.

- Mixin: [`OfflineEncodingMixin`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L31)
- Cache-mode gate: [`OfflineEncodingMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L106)
- Allowed modes per endpoint: [`_ALLOWED_CACHE_MODES`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L25)

## Why cache a module's encoder

An omni training graph often contains modules that are frozen and
deterministic: a VAE encoder that turns images into latents, or a ViT that turns
images into patch features. Nothing trains them, and for the same input they
produce the same output in every epoch and every run. Running them online costs:

- **Repeated compute.** Every training step pays the frozen encoder's forward
  again, although its output never changes.
- **Memory and load time.** Every rank keeps the encoder's weights and
  activations resident and loads them at startup, only to produce tensors that
  could have been read from disk.
- **Coupled layouts.** Encoding needs no optimizer and no backward, yet it has to
  run inside the training job, on the training layout.

Offline caching splits such a module into two stages: an **encode** stage that
runs once, in a dedicated encoding run, and produces a cache artifact per
sample; and a **process** stage that runs every training step on the cached
artifact and produces the tensors the rest of the graph expects.

The cut is two stages rather than "cache the module's output" because not
everything downstream of the encoder is deterministic. A VAE encoder outputs a
posterior (mean and log-variance), and training samples a fresh latent from it
every step. Caching the sampled latent would freeze one sample forever. So the
module cuts at the boundary between the expensive deterministic part and the
cheap per-step part: for a VAE, `offline_encode` stops at the posterior, and
`online_process` samples and scales the latent.

## `support_cache` and `cache_mode`

Whether a module *can* use a cache and whether this run *does* are separate
decisions, so they live in different places:

| Name | Where it lives | Meaning |
|------|----------------|---------|
| `support_cache` | `config.support_cache`, saved in `config.json`; read by [`validate_cache_mode`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L77) | This checkpoint *can* run from an offline cache. |
| `cache_mode` | Constructor kwarg, [kept on the instance](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L88) as `self.cache_mode`; never written to the config | What this run actually does. |

`cache_mode` takes one of three values
([`VALID_CACHE_MODES`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L55)):

| `cache_mode` | Endpoints allowed | Use |
|--------------|-------------------|-----|
| `full` (default) | `offline_encode`, `online_process` | No cache; the module encodes online. `support_cache` is not required. |
| `encode_only` | `offline_encode` | Produce the cache. Requires `support_cache=True`. |
| `process_only` | `online_process` | Train from the cache. Requires `support_cache=True`. |

`model.train_type` is unrelated: it selects the training graph, not the cache
mode.

## How a run picks `cache_mode`

The trainer derives each module's `cache_mode` from the workflow,
`train.train_type`
([`OmniTrainingArguments.module_cache_mode`](../../../veomni/arguments/omni_arguments_types.py#L637)).
Only modules whose config has `support_cache` leave `full`:

| `train.train_type` | `support_cache: true` | otherwise |
|--------------------|-----------------------|-----------|
| `offline_cache` | `encode_only` | `full` |
| `train_with_cache` | `process_only` | `full` |
| `train` | `full` | `full` |

[`ModuleRuntime.cache_mode`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L191)
is the single source of the mode. Inference builds have no train arguments, so
they always use `full`. The same value reaches both sides:

- **Model.** [`ModuleRuntime._build_model`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L282)
  passes it as `build_foundation_model(..., model_kwargs={"cache_mode": ...})`,
  then [raises `ValueError`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L304)
  if the built module ignored it: HF's `PreTrainedModel.__init__` silently drops
  the kwarg on a class that does not mix in `OfflineEncodingMixin`.
- **Data.** [`OmniProcessor.from_config(..., cache_modes=...)`](../../../veomni/models/seed_omni/processing_omni.py#L112)
  and [`bind_module_assets`](../../../veomni/models/seed_omni/modules/module_processing_base.py#L145)
  pass it to the module preprocessor's `from_pretrained`, so a `process_only`
  module can skip CPU-side preprocessing.

## Opting in

Offline encoding is opt-in through multiple inheritance. `BaseMixin` does
**not** inherit `OfflineEncodingMixin`, and modules that do not mix it in are
unaffected. Put the concrete endpoint pair on a sibling `*OfflineMixin`, list
the classes in this order, and write the module's own `@pre_forward` /
`@post_forward` hooks for the two call-sites:

```python
class XxxOfflineMixin:
    def offline_encode(self, pixel_values):
        return {"encoded_cache": ...}

    def online_process(self, encoded_cache):
        return {"latents": ...}


class XxxAccelerated(XxxOfflineMixin, OfflineEncodingMixin, TrainingModuleMixin, BaseMixin, XxxModel):
    # Each hook must return a dict: a pre-hook's dict becomes the endpoint's
    # kwargs, a post-hook's dict is merged into the shared batch.
    @pre_forward("offline_encode")
    def offline_encode_pre(self, conversation_list=None, **batch):
        return {"pixel_values": ...}  # this module's items from conversation_list

    @post_forward("offline_encode")
    def offline_encode_post(self, encoded_cache):
        return {"conversation_list": ...}  # attach each sample's cache artifact

    @pre_forward("online_process")
    def online_process_pre(self, conversation_list=None, **batch):
        return {"encoded_cache": ...}  # cached artifacts read back from the batch

    @post_forward("online_process")
    def online_process_post(self, latents):
        return {"latents": latents}  # where downstream nodes read them
```

```python
# config.json: {"support_cache": true, ...}
model = XxxAccelerated(config, cache_mode="encode_only")
model.cache_mode  # "encode_only"
```

## The functions

### What a module implements or calls

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| [`offline_encode(**kwargs) -> dict`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L99) | **Must implement** on the sibling `*OfflineMixin`. Turn tensor inputs into deterministic cache tensors. | The graph node whose method is `offline_encode`, through [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27) | Produces the cache artifacts. Abstract on the mixin, so a module without it cannot be instantiated. |
| [`online_process(**kwargs) -> dict`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L103) | **Must implement** on the sibling `*OfflineMixin`. Turn cache tensors back into the tensors training needs. | The graph node whose method is `online_process`, through [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27) | Materializes runtime tensors from the cache. Abstract on the mixin, like `offline_encode`. |
| `@pre_forward("offline_encode")`, `@post_forward("offline_encode")`, and the same for `online_process` | **Must write** on the module. Translate between the graph's conversation payload and the endpoint's tensors. The mixin defines none. | [`TrainingModuleMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L59) / [`post_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L66), after the gate | The endpoint receives tensors and the graph receives the hook's output. |
| `cache_mode=` constructor kwarg | Pass `"full"`, `"encode_only"` or `"process_only"` when building the module; `ModuleRuntime` does this from the workflow (see [above](#how-a-run-picks-cache_mode)). | [`OfflineEncodingMixin.__init__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L83) | Selects which endpoints this run may call. Defaults to `full`. |

### What each hook does

The graph passes every node the shared batch and merges whatever the node
returns back into it. The four hooks are where a module translates between that
batch and its tensor endpoints:

| Hook | Receives | Returns | Job |
|------|----------|---------|-----|
| `@pre_forward("offline_encode")` | The shared batch | Kwargs for `offline_encode`, e.g. `{"pixel_values": ...}` | Select this module's items from the batch and stack them into input tensors. |
| `@post_forward("offline_encode")` | `offline_encode`'s outputs | A dict merged into the batch | Attach each sample's cache artifact to the batch, e.g. onto its conversation item, where a cache writer can persist it. |
| `@pre_forward("online_process")` | The shared batch, carrying cached artifacts | Kwargs for `online_process`, e.g. `{"encoded_cache": ...}` | Read the cached artifacts back out of the batch and move them to the device. |
| `@post_forward("online_process")` | `online_process`'s outputs | A dict merged into the batch | Write the results where downstream nodes read them, i.e. the same place the online encode path writes. |

`offline_encode` takes the same inputs as the module's online encode method, and
`online_process` returns the same outputs, so one hook can usually serve both
call-sites: `@pre_forward("encode", "offline_encode")` and
`@post_forward("encode", "online_process")`. Downstream nodes then cannot tell
whether the module ran online or from the cache.

### What the framework provides (do not override)

| Function | Called by | Effect |
|----------|-----------|--------|
| [`OfflineEncodingMixin.__init__(*args, cache_mode="full", **kwargs)`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L83) | Module construction, before the native model body | Validates `cache_mode` against the config, then [sets `self.cache_mode`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L88) **before** [`super().__init__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L89) runs the model body, so the body can skip sub-networks the mode never uses. For example, a VAE in `encode_only` need not allocate its decoder. |
| [`OfflineEncodingMixin.validate_cache_mode(cache_mode, config)`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L70) | `__init__`, once before and, if the config only appears on `self.config`, once after the model body | Raises `ValueError` for an unknown mode, or for `encode_only` / `process_only` on a config without `support_cache`. |
| [`OfflineEncodingMixin.pre_forward(method, **kwargs)`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L106) | [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L63) on the graph path, and [`OmniModel._run_train_node`](../../../veomni/models/seed_omni/modeling_omni.py#L454) on the eager path | Checks `method` against [`_ALLOWED_CACHE_MODES`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L25) and [raises `ValueError`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L111) before the module's own pre-hook runs, then [hands off](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L115) to `TrainingModuleMixin.pre_forward`. Other methods pass straight through. |
| [`OfflineEncodingMixin.__init_subclass__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L60) | Python, when a subclass is defined | Raises `TypeError` if `TrainingModuleMixin` comes before `OfflineEncodingMixin` in the MRO. |

## Lifecycle

- Construction:
  [`ModuleRuntime` → `build_foundation_model`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L295) →
  the module class
  - [`OfflineEncodingMixin.__init__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L83):
    [validate](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L70) `cache_mode`, then
    [set `self.cache_mode`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L88).
  - [`super().__init__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L89): the native model body,
    which may read `self.cache_mode`.
- Each `offline_encode` / `online_process` node:
  [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27)
  - [`raw.pre_forward(method, **batch)`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L63)
    - [`OfflineEncodingMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L106): the cache-mode gate.
    - [`TrainingModuleMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L59): the module's `@pre_forward(method)` hook.
  - [Endpoint](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L68): the sibling mixin's `offline_encode` / `online_process`, then `post_forward`.

## Resulting workflow

Offline caching turns one training job into two runs over the same data, each
with its own graph. The endpoint is named in the graph as `module.method`:

```yaml
# Encoding run, cache_mode: encode_only
- {from: xxx.offline_encode, to: end}
```

```yaml
# Training run, cache_mode: process_only
- {from: xxx.online_process, to: yyy}   # replaces {from: xxx.encode, to: yyy}
- {from: yyy, to: end}
```

1. **Encoding run.** The module is built without the sub-networks it does not
   need for encoding, e.g. a VAE skips its decoder. Only `offline_encode` runs,
   without autograd. Its post-hook attaches the artifacts to the batch, and a
   cache writer persists them per sample.
2. **Training run.** The dataset yields the cached artifacts in place of the raw
   media for this module. The module is built without its encoder, and
   `online_process` turns each artifact into the tensors downstream nodes expect.

The result:

- The encoder runs once per dataset instead of once per step in every run.
- Training ranks hold neither the encoder's weights nor its activations, and do
  not load them at startup.
- The training data pipeline no longer decodes and preprocesses the raw media
  for this module.
- Per-step randomness after the cut, such as VAE latent sampling, is kept,
  because `online_process` still runs every step.
- A graph that calls an endpoint the module's mode cannot serve fails on the
  first step instead of silently training on the wrong path.

The pieces that complete this workflow are in place:
[`OmniTrainer.offline_cache_step`](../../../veomni/trainer/omni/omni_trainer.py#L540)
runs the encoding run, and
[`SeedOmniOfflineCacheWriter.save_conversation_list`](../../../veomni/models/seed_omni/utils/offline_cache.py#L92)
persists each conversation. The training run reads the cache through the
[`seedomni_cached`](../../../veomni/data/seed_omni/seedomni_transform.py#L246)
data transform. See the
[Bagel offline VAE cache](../example_models/bagel.md#32-offline-vae-posterior-cache-two-stages)
for a two-stage example.

## Rules for a correct implementation

- **`XxxOfflineMixin` goes before `OfflineEncodingMixin`,** so the concrete
  endpoints win MRO lookup over the abstract ones.
- **`OfflineEncodingMixin` goes before `TrainingModuleMixin`.**
  [`TrainingModuleMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L59)
  does not call `super()`, so listing it first would skip the cache-mode gate.
  [`__init_subclass__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L60)
  rejects such a class with `TypeError` when it is defined.
- **Read `self.cache_mode`, not the config,** to decide which sub-networks to
  build. `cache_mode` is per run and is never saved to `config.json`.
- **Call endpoints through `pre_forward`.** The gate only covers calls dispatched
  through `pre_forward`, which is what the omni graph does. Calling a concrete
  endpoint directly bypasses it.

## Current scope

- **Reduced modes are not checkpointed.** A module in `encode_only` or
  `process_only` lacks sub-networks, so it must be fully frozen;
  [`ModuleRuntime._check_cache_mode_is_frozen`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L368)
  raises `ValueError` for a trainable one.
- **No per-module checkpoint hooks.** Checkpoint I/O stays with
  [`OmniModuleCheckpointManager`](../../../veomni/models/seed_omni/utils/checkpoint.py#L30).
  [Fully frozen modules have no checkpoint manager](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L593),
  so they skip it.

## Tests

- [`tests/seed_omni/mixins/test_offline_encoding_mixin.py`](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py):
  - [`cache_mode` validation](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L74) and the
    [`support_cache` requirement](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L87);
  - the [cache-mode gate](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L93);
  - [`cache_mode` set before the model body and kept off the config](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L119);
  - [MRO order of the sibling mixin](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L149) and
    [rejection of the wrong base order](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L178).
- [`tests/seed_omni/runtime/test_module_runtime.py`](../../../tests/seed_omni/runtime/test_module_runtime.py):
  - [workflow → `cache_mode` → constructor kwarg](../../../tests/seed_omni/runtime/test_module_runtime.py#L155);
  - [a module that ignored its `cache_mode`](../../../tests/seed_omni/runtime/test_module_runtime.py#L182) and
    [a trainable module in a reduced mode](../../../tests/seed_omni/runtime/test_module_runtime.py#L197) are rejected.
- [`tests/seed_omni/model/test_processor.py`](../../../tests/seed_omni/model/test_processor.py):
  the [processor](../../../tests/seed_omni/model/test_processor.py#L213) and
  [`bind_module_assets`](../../../tests/seed_omni/model/test_processor.py#L234) forward a non-`full` mode.
- [`tests/seed_omni/bagel/test_processing.py`](../../../tests/seed_omni/bagel/test_processing.py):
  [Bagel VAE built through `build_foundation_model`](../../../tests/seed_omni/bagel/test_processing.py#L379) and the
  [offline cache round trip](../../../tests/seed_omni/bagel/test_processing.py#L319).
