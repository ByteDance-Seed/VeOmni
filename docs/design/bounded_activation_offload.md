# Bounded checkpoint activation offload

The bounded backend saves selected checkpoint inputs through four fixed pinned
staging buffers and separate pageable snapshots. Unlike the direct async
backend's idle-cache limit, these budgets limit the staging allocations and
live snapshots owned by one manager. Intermediate activations still use normal
gradient-checkpoint recomputation.

```yaml
model:
  accelerator:
    gradient_checkpointing:
      enable: true
      enable_reentrant: false
      recompute_layers: -1
    offload_config:
      enable_async_activation: true
      activation_offload_backend: bounded
      activation_offload_modules: ["model.layers.{*}"]
      activation_offload_staging_buffer_mib: 192
      activation_offload_pageable_limit_gb: 8
      activation_offload_prefetch_depth: 1
      activation_offload_pipeline_transfers: false
```

The default backend remains `direct`. Bounded offload requires explicit target
modules using full nonreentrant checkpointing. Each target must expose a
contiguous `hidden_states` tensor as its first positional argument. Selective
or partial recomputation is not supported by this backend. Ordinary attention
metadata arguments are passed through unchanged.

## Transfer and memory ownership

Two D2H and two H2D slots are allocated lazily. For a configured slot size of
`B` bytes, the requested pinned staging capacity is `4 * B`; it does not grow
with layer count or gradient accumulation. A checkpoint input larger than one
slot is rejected. The pageable limit separately counts live immutable activation
snapshots and fails admission rather than waiting for a backward that cannot
start. CPU snapshots remain owned by autograd until their saved tensors die.

The producer stream records completion before a D2H worker may read its input.
Snapshots protect the saved values if the producer later reuses the source.
H2D restores publish a completion event; the consuming stream waits on it and
records the restored allocation's stream ownership. Pinned slots are reused
only after DMA completion. Worker failures are sticky, and partially submitted
copies retain their payload rather than freeing storage still in use.

Reverse prefetch may restore the preceding checkpoint while the current one is
consumed. Only one speculative restore is admitted, leaving a credit for demand
to avoid starvation. `activation_offload_pipeline_transfers: true` separates
DMA submission from completion bookkeeping while retaining the same four slots.

These are per-manager requested-byte bounds. Consumer-owned device tensors,
allocator rounding/caches, other managers and the rest of the training process
are outside them. Whole-host/cgroup memory policy belongs to the runtime or
job configuration; this backend does not require a finite container limit.

## Framework integration and API

[`VeOmniModelRuntime._apply_async_activation_offload()`](../../veomni/models/model_runtime.py#L341)
installs the selected
backend before FSDP wrapping. It calls `apply_bounded_activation_offload()`
with module patterns and a
[`StagingConfig`](../../veomni/distributed/bounded_activation_offload.py#L41).
No trainer subclass or model
forward implementation needs an offload-specific override.

| API | Owner and effect |
|---|---|
| `StagingConfig` | Framework-provided configuration of slot size, live-snapshot budget, prefetch and transfer pipeline. |
| `apply_bounded_activation_offload(model, patterns, config)` | Runtime calls once before FSDP; wraps selected checkpoint boundaries and returns the manager. |
| `pack` / `unpack` | Framework-provided saved-tensor hooks; schedule snapshot and restore work. |
| `reset()` | Existing runtime offload-manager lifecycle calls between steps; checks health without clearing live autograd snapshots. |
| `snapshot_stats()` | Optional observation of manager-owned bytes, ticket counts, waits and prefetch counts. |
| `drain()` | Optional quiescent test/inspection boundary waiting for queued bookkeeping; not required in the training hot path. |
| `close()` | Explicit owner/test teardown; waits for workers before releasing staging storage. Model finalization also releases its ownership while live graphs retain theirs. |

The backend does not change model values, labels, loss normalization, optimizer
precision, Ulysses/CP or the checkpoint format. It is independent of selective
expert and optimizer-state CPU offload.
