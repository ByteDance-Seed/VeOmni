# Offline Encoding for Omni Modules

`OfflineEncodingMixin` (exported from `veomni.models.seed_omni`) lets an omni
module run an expensive, deterministic encoder, such as a VAE or a ViT, once
offline, and train from the cached tensors afterwards. A module opts in through
multiple inheritance; modules that never mix it in are unaffected.

## `support_cache` and `cache_mode`

| Name | Where it lives | Meaning |
|------|----------------|---------|
| `support_cache` | `config.support_cache`, saved in `config.json` | This checkpoint *can* run from an offline cache. |
| `cache_mode` | Constructor kwarg, kept on the instance as `self.cache_mode`; never written to the config | What this run actually does. |

`cache_mode` takes one of three values:

| `cache_mode` | Endpoints allowed | Use |
|--------------|-------------------|-----|
| `full` (default) | `offline_encode`, `online_process` | No cache; the module encodes online. `support_cache` is not required. |
| `encode_only` | `offline_encode` | Produce the cache. Requires `support_cache=True`. |
| `process_only` | `online_process` | Train from the cache. Requires `support_cache=True`. |

An unknown mode, or `encode_only` / `process_only` on a config without
`support_cache`, raises `ValueError` at construction.

`OfflineEncodingMixin.__init__` sets `self.cache_mode` **before** the native
model body runs, so the body can skip sub-networks the mode never uses. For
example, a VAE in `encode_only` need not allocate its decoder.

`train_type` is unrelated: it selects the training graph, not the cache mode.

## Hook contract

A module provides two tensor endpoints. They are abstract on the mixin:

```python
def offline_encode(self, **kwargs) -> dict[str, Any]:
    """Produce deterministic tensor cache artifacts from tensor inputs."""

def online_process(self, **kwargs) -> dict[str, Any]:
    """Materialize runtime tensors from offline encoded cache tensors."""
```

Put the concrete pair on a sibling `*OfflineMixin` and list the classes in this
order:

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

The base order has two requirements:

- **`XxxOfflineMixin` goes before `OfflineEncodingMixin`,** so that the concrete
  endpoints win MRO lookup over the abstract ones.
- **`OfflineEncodingMixin` goes before `TrainingModuleMixin`.**
  `TrainingModuleMixin.pre_forward` does not call `super()`, so listing it
  first would skip the cache-mode gate. Such a class fails at definition time
  with `TypeError`.

The `@pre_forward` / `@post_forward` hooks for `offline_encode` and
`online_process` are the module's own: they translate between the graph's
conversation payload and the endpoint's tensors. The mixin defines none.

## Cache-mode gate

`OfflineEncodingMixin.pre_forward` checks the table above on every call. If
`offline_encode` is dispatched in `process_only`, or `online_process` in
`encode_only`, it raises `ValueError` before the module's own pre-hook runs.
The gate covers calls dispatched through `pre_forward`, the path the omni graph
uses. Calling a concrete endpoint directly bypasses it.

## Current scope

- **Launcher configs don't set `cache_mode` yet.** `build_foundation_model`
  takes no model kwargs, so modules built by the runtime always run in `full`.
  Passing `cache_mode` from the launcher config, writing cache shards, and
  loading state dicts in the reduced modes are planned for a follow-up.
- **No per-module checkpoint hooks.** Checkpoint I/O stays with
  `OmniModuleCheckpointManager`: fully frozen modules have no checkpoint
  manager, so they skip it.

Source: [`veomni/models/seed_omni/mixins/offline_encoding_mixin.py`](../../veomni/models/seed_omni/mixins/offline_encoding_mixin.py).
Tests: [`tests/seed_omni/mixins/test_offline_encoding_mixin.py`](../../tests/seed_omni/mixins/test_offline_encoding_mixin.py).
