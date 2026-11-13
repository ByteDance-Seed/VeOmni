# Data loader resume protocol

`GlobalStateCallback` owns the training position. When restoring a mid-epoch
checkpoint, it computes `start_epoch` and `start_step` from the saved step and
calls the optional `train_dataloader.prepare_resume_epoch(start_epoch)` before
`train_dataloader.load_state_dict(...)`. The hook is not called for an
epoch-boundary restore. The checkpoint and loader restore only the cursor and
RNG state they actually save; the hook supplies an epoch that an older data
snapshot may be missing.

## Implementing the hook

A loader or dataset wrapper that contains an epoch-aware source should forward
`prepare_resume_epoch(epoch)` to that source if it implements the hook. The
built-in `DynamicBatchSizeDataLoader`, `DistributedDataloader`, and
`DynamicBatchingSizeDataset` forward it to their wrapped loader or dataset.
Wrappers with no such source may omit the hook. Forwarding is especially
important when `load_state_dict()` starts workers or prefetches immediately.

For example, this fragment shows how a custom wrapper can forward the optional
hook without assuming that every wrapped loader implements it. A real loader
also needs its usual iteration, save, and other delegation methods:

```python
class LoaderWrapper:
    def __init__(self, inner):
        self.inner = inner

    def prepare_resume_epoch(self, epoch):
        prepare = getattr(self.inner, "prepare_resume_epoch", None)
        if callable(prepare):
            prepare(epoch)

    def load_state_dict(self, state):
        return self.inner.load_state_dict(state)
```

The built-in restore path is:

1. [`GlobalStateCallback.load_global_state()`](../../veomni/trainer/callbacks/global_state_callback.py#L244)
   computes the saved position and calls the hook for a mid-epoch restore.
2. The optional [main-process dynamic batcher](../../veomni/data/dynamic_batching.py#L413)
   forwards the hook to the [distributed loader](../../veomni/data/data_loader.py#L55),
   which forwards it to its dataset.
3. The optional [worker-side batching dataset](../../veomni/data/dataset.py#L1571)
   forwards it to the [multi-source dataset](../../veomni/data/dataset.py#L719).
4. The callback loads the loader state and then restores the saved trainer RNG
   state. The multi-source dataset consumes the supplied epoch when it
   [loads legacy state](../../veomni/data/dataset.py#L914); depending on the
   loader, that can happen during the load or when iteration starts later.

The hook must not advance or reset a source cursor, create an iterator, consume
samples, or alter RNG state. It must not replace an epoch already present in a
checkpoint. Normal `load_state_dict()` restores the cursor and sampling state;
the trainer restores its saved RNG state separately. A wrapper must preserve
the hook-before-load order when it delegates restoration.

`WeightedMultiSourceDataset` records the supplied epoch temporarily. A modern
snapshot's saved epoch takes precedence. For a legacy snapshot without an epoch,
the supplied value identifies the restored sampling epoch without reseeding the
restored RNG or resetting source cursors. If the epoch is unknown and sampling
has already continued, a later `set_epoch()` is rejected instead of silently
using a possibly wrong epoch. The temporary value is not itself a checkpoint or
a replacement for the source's `state_dict()` / `load_state_dict()`.

Regression coverage is in `tests/data/test_multisource_rng_resume.py`: the
[main-process batching](../../tests/data/test_multisource_rng_resume.py#L209)
case exercises hook forwarding through the trainer callback, while the
[loader resume](../../tests/data/test_multisource_rng_resume.py#L160) case
checks delayed `set_epoch()` with and without worker-side batching. The
[second snapshot](../../tests/data/test_multisource_rng_resume.py#L313) and
[unknown epoch](../../tests/data/test_multisource_rng_resume.py#L356) cases cover
continued sampling after legacy restoration. Run them with
`pytest -q tests/data/test_multisource_rng_resume.py`.

## Scope

This protocol cannot reconstruct a legacy worker's missing RNG state or
identify arbitrary custom transformations' random state. A legacy snapshot may
contain `random_state` but lack `sampling_initialized`: in that case the
loader assumes sampling had started. If a worker was saved before its first
sample, it may choose a different source after restore even when its epoch is
known. Current snapshots record the initialization flag, but it cannot be
inferred reliably from an older snapshot's RNG state alone.

An upstream Hugging Face iterable may not save its shuffle buffer or epoch,
and eager worker prefetch can begin before the upstream epoch is restored.
Exact continuation therefore depends on the actual source and loader
configuration.

For multi-epoch `WeightedMultiSourceDataset` runs, use
`persistent_workers=False`: a persistent worker's dataset copy may otherwise
continue sampling with the previous epoch. Keeping workers alive alone does not
propagate a new epoch to each worker-local source.
