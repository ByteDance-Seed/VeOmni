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

## Scope

This protocol cannot reconstruct a legacy worker's missing RNG state or
identify arbitrary custom transformations' random state. An upstream Hugging
Face iterable may not save its shuffle buffer or epoch, and eager worker
prefetch can begin before the upstream epoch is restored. Exact continuation
therefore depends on the actual source and loader configuration.

For multi-epoch `WeightedMultiSourceDataset` runs, use
`persistent_workers=False`: a persistent worker's dataset copy may otherwise
continue sampling with the previous epoch. Keeping workers alive alone does not
propagate a new epoch to each worker-local source.
