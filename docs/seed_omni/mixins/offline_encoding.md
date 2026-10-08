# Offline Encoding (`OfflineEncodingMixin`)

`OfflineEncodingMixin` lets a SeedOmni module run an expensive, deterministic
encoder, such as a VAE or a ViT, once offline, and train from the cached tensors
afterwards.

- Mixin: [`OfflineEncodingMixin`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L31)
- Cache-mode gate: [`OfflineEncodingMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L106)
- Allowed modes per endpoint: [`_ALLOWED_CACHE_MODES`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L25)

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

`train_type` is unrelated: it selects the training graph, not the cache mode.

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
    @pre_forward("offline_encode")
    def offline_encode_pre(self, conversation_list=None, **batch):
        ...  # conversation payload -> offline_encode kwargs

    @post_forward("offline_encode")
    def offline_encode_post(self, **outputs):
        ...

    @pre_forward("online_process")
    def online_process_pre(self, conversation_list=None, **batch):
        ...

    @post_forward("online_process")
    def online_process_post(self, **outputs):
        ...
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
| `cache_mode=` constructor kwarg | Pass `"full"`, `"encode_only"` or `"process_only"` when building the module. | [`OfflineEncodingMixin.__init__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L83) | Selects which endpoints this run may call. Defaults to `full`. |

### What the framework provides (do not override)

| Function | Called by | Effect |
|----------|-----------|--------|
| [`OfflineEncodingMixin.__init__(*args, cache_mode="full", **kwargs)`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L83) | Module construction, before the native model body | Validates `cache_mode` against the config, then [sets `self.cache_mode`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L88) **before** [`super().__init__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L89) runs the model body, so the body can skip sub-networks the mode never uses. For example, a VAE in `encode_only` need not allocate its decoder. |
| [`OfflineEncodingMixin.validate_cache_mode(cache_mode, config)`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L70) | `__init__`, once before and, if the config only appears on `self.config`, once after the model body | Raises `ValueError` for an unknown mode, or for `encode_only` / `process_only` on a config without `support_cache`. |
| [`OfflineEncodingMixin.pre_forward(method, **kwargs)`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L106) | [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L63) on the graph path, and [`OmniModel._run_train_node`](../../../veomni/models/seed_omni/modeling_omni.py#L454) on the eager path | Checks `method` against [`_ALLOWED_CACHE_MODES`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L25) and [raises `ValueError`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L111) before the module's own pre-hook runs, then [hands off](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L115) to `TrainingModuleMixin.pre_forward`. Other methods pass straight through. |
| [`OfflineEncodingMixin.__init_subclass__`](../../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py#L60) | Python, when a subclass is defined | Raises `TypeError` if `TrainingModuleMixin` comes before `OfflineEncodingMixin` in the MRO. |

## Lifecycle

- Construction:
  [`OmniModuleRuntime` → `build_foundation_model`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L259) →
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

- **Launcher configs don't set `cache_mode` yet.**
  [`OmniModuleRuntime` calls `build_foundation_model`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L259)
  with no model kwargs, so modules built by the runtime always run in `full`.
  Passing `cache_mode` from the launcher config, writing cache shards, and
  loading state dicts in the reduced modes are planned for a follow-up.
- **No per-module checkpoint hooks.** Checkpoint I/O stays with
  [`OmniModuleCheckpointManager`](../../../veomni/models/seed_omni/utils/checkpoint.py#L30).
  [Fully frozen modules have no checkpoint manager](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py#L533),
  so they skip it.

## Tests

- [`tests/seed_omni/mixins/test_offline_encoding_mixin.py`](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py):
  - [`cache_mode` validation](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L74) and the
    [`support_cache` requirement](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L87);
  - the [cache-mode gate](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L93);
  - [`cache_mode` set before the model body and kept off the config](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L119);
  - [MRO order of the sibling mixin](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L149) and
    [rejection of the wrong base order](../../../tests/seed_omni/mixins/test_offline_encoding_mixin.py#L178).
