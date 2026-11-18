# Exact data-resume extensions

The [checkpoint guide](checkpoint.md) describes the built-in exact-resume
scope and the required input, worker, and topology attestations. A custom source
or loader is responsible for its own deterministic replay and validation;
implementing the interface alone does not verify it.

## Weighted sources

Each custom source used in a `WeightedMultiSourceDataset` needs these members:

- `exact_resume_fingerprint: str`: a stable identity for the immutable data
  snapshot, source implementation and configuration. A changed input or
  behavior must change this string. The value is incorporated into the
  multi-source topology fingerprint, not independently validated against files.
- `state_dict()` and `load_state_dict(state)`: serialize and restore the full
  source cursor and any state that affects sample order, including its RNG.
- `validate_exact_resume_state(state, *, worker_geometry, epoch)`: check the
  saved cursor, snapshot identity, worker geometry `(worker_id, num_workers)`
  and epoch before loading. Raise on a mismatch or malformed state. Validation
  must not mutate the source, consume data or advance RNG.
- For persistent-worker epoch transitions, `supports_exact_epoch_restart`
  must be true and `reset_for_epoch(epoch)` must rebuild that epoch's initial
  stream independently of the preceding epoch. It must work when called more
  than once with the same epoch.

`WeightedMultiSourceDataset` gathers source identities during initialization,
validates each source in `validate_exact_resume_state`, and calls
`reset_for_epoch` when entering a persistent-worker epoch. The implementation
currently declares persistent-worker **cold** resume only for its built-in
replayable HF streams; the custom-source members do not lift that restriction.

An outline for a deterministic, immutable source is:

```python
class SnapshotSource:
    exact_resume_fingerprint = "dataset-v1-and-transform-digest"
    supports_exact_epoch_restart = True

    def state_dict(self):
        return {"snapshot": self.exact_resume_fingerprint, "epoch": self.epoch,
                "worker": self.worker_geometry, "cursor": self.cursor}

    def validate_exact_resume_state(self, state, *, worker_geometry, epoch):
        if (state["snapshot"], state["worker"], state["epoch"]) != (
            self.exact_resume_fingerprint, worker_geometry, epoch
        ):
            raise ValueError("source snapshot, worker, or epoch changed")
        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= self.size:
            raise ValueError("invalid source cursor")

    def load_state_dict(self, state):
        # Called only after the loader has validated the state.
        self.epoch, self.cursor = state["epoch"], state["cursor"]

    def reset_for_epoch(self, epoch):
        self.epoch, self.cursor = epoch, 0
```

Real implementations also need iteration and must include shuffle state,
sharding, worker identity and any stateful transforms in the saved cursor and
fingerprint. Use the same worker and DP geometry on restore.

## Loader with a prepared resume iterator

For a mid-epoch resume of a native map-style `DistributedDataloader` using
the standard batch sampler, no custom loader generator, and either
`StatefulDistributedSampler` or PyTorch's `DistributedSampler`, the trainer
sets the restored sampler epoch before
workers prefetch, then retains the restored iterator before restoring its CPU
RNG. It also retains a saved iterator at step zero; an empty step-zero cursor
instead starts normally. The same RNG ordering applies to the replayable
multi-source loader under its validated source contract. At an epoch boundary,
the standard `StatefulDistributedSampler` starts from the beginning of the
new epoch rather than reusing the previous epoch's cursor. The normal
new-epoch iterator is still created after RNG restoration.

The locked runtime uses `torchdata` 0.11. With the older 0.9/0.10 batch
sampler, restoring a nonzero-step plain `DistributedSampler` cursor is
rejected because the sampler epoch cannot be proven. A native fixed loader
with `StatefulDistributedSampler` and workers is also rejected at an epoch
boundary on those versions, as is a dynamic loader with workers when the
runtime cannot establish ordered delivery. Use the locked runtime for these
cases. Saving again before a restored zero-worker sampler advances preserves
its pending cursor; an unrecognized sampler snapshot fails rather than
writing a cursor that may replay samples.

`GlobalStateCallback.load_global_state` normally primes `iter(loader)` for a
non-strict loader resumed at an epoch boundary. It primes the supported native
loaders described above before restoring the CPU RNG, and other loaders after
restoring it. A custom stateful loader that
already creates a restored iterator in `load_state_dict` can set
`_veomni_resume_iterator_prepared = True` **after** installing its restored
queue. The callback then skips that extra priming call. The subsequent training
loop's first `iter(loader)` must return that same prepared iterator without
resetting or discarding the queue. Clear the flag when taking the iterator and
when closing or resetting the loader; do not set it before the queue is ready.

```python
def load_state_dict(self, state):
    self._resume_iterator = self.build_restored_iterator(state)
    self._veomni_resume_iterator_prepared = True

def __iter__(self):
    if self._veomni_resume_iterator_prepared:
        self._veomni_resume_iterator_prepared = False
        return self._resume_iterator
    return self.build_new_iterator()
```

This flag only avoids duplicate initialization. It does not certify custom
loader ordering, epoch boundaries, worker prefetch, or cold recovery; validate
those separately before relying on exact resume. Source identity, stateful
transforms and custom generator state require their own verified contract.
