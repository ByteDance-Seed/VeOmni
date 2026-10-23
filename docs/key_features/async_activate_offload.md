# Async Activation Offload

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [Motivation](#motivation)
  - [Stream-Based D2H/H2D Overlap](#stream-based-d2hh2d-overlap)
  - [Interaction with Gradient Checkpointing](#interaction-with-gradient-checkpointing)
  - [Automatic Module Discovery](#automatic-module-discovery)
- [Configuration Reference](#configuration-reference)
- [Core API](#core-api)
- [Implementation Details](#implementation-details)
  - [SwapTensor Lifecycle](#swaptensor-lifecycle)
  - [OffloadManager Ownership](#offloadmanager-ownership)
  - [Per-Instance Class Patching](#per-instance-class-patching)
  - [Prefetch Mechanism](#prefetch-mechanism)
- [Limitations](#limitations)

## Overview

Async activation offload is a memory optimization feature that asynchronously transfers
activation tensors from GPU to CPU (D2H) during the forward pass and prefetches them
back (H2D) during the backward pass, using dedicated CUDA/NPU streams. This overlaps
memory transfers with computation, reducing GPU memory consumption with minimal
throughput impact compared to synchronous activation offload.
The feature is designed for large-model training scenarios where GPU memory is the
primary bottleneck. By offloading hidden states to CPU and relying on gradient
checkpointing (recomputation) for intermediate activations, it enables training with
significantly reduced peak GPU memory.

**Key benefits:**
- Reduces peak GPU memory by offloading hidden states to CPU
- Overlaps D2H/H2D transfers with computation via dedicated streams
- Prefetches activations for the next layer during backward pass
- Auto-discovers offload targets from `model._no_split_modules`

## Quick Start

To enable async activation offload, set the
`model.accelerator.offload_config.enable_async_activation` parameter in your
configuration file:

```yaml
model:
  accelerator:
    gradient_checkpointing:
      enable: true            # Recommended for lower activation memory
    offload_config:
      enable_async_activation: true

```


Or via command-line:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3-moe.yaml \
    --model.accelerator.gradient_checkpointing.enable true \
    --model.accelerator.offload_config.enable_async_activation true

```

**Recommendation:** `gradient_checkpointing.enable` should be `True` when using async
activation offload. The feature relies on recomputation to avoid storing intermediate
activations on GPU. Without gradient checkpointing, hidden states are still offloaded
but intermediate activations within each block persist on GPU, reducing the memory
savings. No runtime error is raised if checkpointing is disabled.

## How It Works

### Motivation

During standard training with gradient checkpointing, hidden states (input activations
to each transformer block) are still saved on GPU for the backward pass. For large
models, these hidden states can consume significant GPU memory.

Async activation offload addresses this by:
1. **Forward pass:** Immediately offloading hidden states to CPU via a dedicated stream
   (non-blocking D2H copy with pinned memory).
2. **Backward pass:** Prefetching hidden states back to GPU just before they are needed
   (H2D copy overlapping with ongoing computation).
3. **Recomputation:** Intermediate activations within each block are recomputed by
   gradient checkpointing, so they never persist on GPU.

This combination reduces persistent activation memory. The final block stays resident,
and tensors that fail the storage or device eligibility checks are not offloaded.

### Stream-Based D2H/H2D Overlap

Unlike the synchronous `custom_save_on_cpu` (which blocks the default stream until
the transfer completes), async activation offload uses a dedicated swap stream:

```text
Default Stream:  [Compute Layer 0] [Compute Layer 1] [Compute Layer 2] ...
Swap Stream:     [D2H Layer 0]     [D2H Layer 1]     [D2H Layer 2]    ...
                                       ↑ overlapped with compute

```

During the forward pass, after computing each layer's forward, the hidden states are
copied to CPU on the swap stream without blocking the default stream. The GPU storage
is then freed (after synchronization), reducing peak memory.
During the backward pass, before each layer's backward needs its hidden states, they
are prefetched back to GPU on the swap stream.

### Interaction with Gradient Checkpointing

The `saved_tensors_hooks` stack ordering is critical. Per-instance subclass `__call__` patching
places `async_save_on_cpu` **outside** the `GradientCheckpointingLayer` checkpoint
boundary:

```text
Forward saved_tensors_hooks stack:
  [async_save_on_cpu (outer)]   ← intercepts hidden_states
  [_checkpoint_hook (inner)]    ← handles intermediate activations via recomputation

```

With this ordering:
- `_NoopSaveInputs.apply` (called by checkpoint before `_checkpoint_hook` is pushed)
  saves input tensors through `async_save_on_cpu.pack`, allowing hidden states to be
  offloaded to CPU.
- Intermediate activations are handled by `_checkpoint_hook` (GC recomputation),
  requiring zero persistent GPU memory.

This ordering requires checkpointing to occur inside the selected module's
`__call__`, as in `GradientCheckpointingLayer`. Wan instead invokes checkpointing
outside the block call. Enabling async offload there places the offload hooks
inside the checkpoint hooks and suppresses recomputation. The Wan I2V examples
therefore keep gradient checkpointing enabled and async offload disabled; an
integration around the outer checkpoint call is needed before combining them.

### Automatic Module Discovery

Offload targets are automatically derived from `model._no_split_modules` by default.
Models with additional FSDP wrapping boundaries can set
`_activation_offload_target_classes` to a separate list of class names, so
offload does not also wrap parent modules. Explicit
`activation_offload_modules` patterns take precedence over auto-discovery.
For example, with Qwen3-MoE where `_no_split_modules = ["Qwen3MoeDecoderLayer"]`,
all `Qwen3MoeDecoderLayer` instances are automatically identified as offload targets.
See the [model target contract](async_activation_offload_targets.md) for the
optional override and its interaction with FSDP wrapping.

## Configuration Reference

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model.accelerator.offload_config.enable_async_activation` | bool | `False` | Enable async activation offload |
| `model.accelerator.offload_config.enable_activation` | bool | `False` | Enable synchronous activation offload (legacy) |
| `model.accelerator.offload_config.activation_gpu_limit` | float | `0.0` | GPU memory limit (GB) for synchronous offload |
| `model.accelerator.gradient_checkpointing.enable` | bool | `True` | Recommended for lower activation memory |

`enable_async_activation` and `enable_activation` are mutually exclusive; enabling
both raises a configuration error. In async mode, the global forward/backward offload
contexts become `nullcontext` because transfers are handled per-module via
`saved_tensors_hooks`.

## Core API

### `apply_async_activation_offload(model, activation_offload_modules, ...)`

Apply async activation offload to matched submodules. Called automatically by
`VeOmniModelRuntime._build_parallelized_model` before sharding when the offload
config sets `enable_async_activation=True`.

```python
from veomni.distributed.async_offload import apply_async_activation_offload
apply_async_activation_offload(model, activation_offload_modules=[])

```

Args:
- `model`: The model to apply offloading to.
- `activation_offload_modules`: Explicit module-name patterns, or `[]` to discover
  targets from `model._activation_offload_target_classes` when defined, otherwise
  from `model._no_split_modules`. Missing or unmatched targets raise an error.
- `host_cache_limit_bytes`: Optional idle pinned-buffer cache budget for this call.
- `host_buffer_pool`: Optional shared `PinnedBufferPool` for multiple model roots;
  mutually exclusive with `host_cache_limit_bytes`.

### `async_save_on_cpu(block_idx, depth, manager, custom_check_fn=None, prefetch=True)`

A `saved_tensors_hooks` context that intercepts saved tensors and offloads them to
CPU. The high-level API supplies one manager per matched module sequence. Manual
callers must create and reuse a manager for all blocks in the same sequence:

```python
from veomni.distributed.async_offload import OffloadManager, PinnedBufferPool, async_save_on_cpu

pool = PinnedBufferPool()
manager = OffloadManager(host_buffer_pool=pool)
try:
    for block_idx, block in enumerate(blocks):
        with async_save_on_cpu(block_idx=block_idx, depth=len(blocks), manager=manager):
            hidden_states = block(hidden_states)
    loss_fn(hidden_states).backward()
finally:
    manager.reset()
```

Args:

- `block_idx`: Index of the current module in the offload sequence.
- `depth`: Total number of modules being offloaded.
- `manager`: Required `OffloadManager` shared by the blocks in this sequence.
- `custom_check_fn`: Optional function to filter which tensors to offload.
- `prefetch`: Whether to prefetch the next layer's activations during backward.

## Implementation Details

### SwapTensor Lifecycle

Each tensor that is offloaded is wrapped in a `SwapTensor` object that manages its
lifecycle across GPU and CPU:

1. **Creation:** A pinned CPU buffer (`tensor_cpu`) is allocated matching the tensor's
   shape and dtype.
2. **D2H (launch_d2h):** On the swap stream, copy GPU tensor to pinned CPU buffer
   (non-blocking). Record a `d2h_event` for synchronization.
3. **Wait D2H (wait_d2h_finished):** After the D2H transfer completes, free the GPU
   storage by resizing to 0.
4. **H2D (launch_h2d):** On the swap stream, resize GPU storage back and copy from
   pinned CPU buffer (non-blocking). Record an `h2d_event`.
5. **Wait H2D:** The default stream waits on `h2d_event` before accessing the tensor.

State transitions: `device` → (D2H) → `host` → (H2D) → `device`

### OffloadManager Ownership

`OffloadManager` owns the bookkeeping and swap stream for one matched module
sequence. It is not a singleton. Each call to `async_offload_modules` creates a
manager and receives a `PinnedBufferPool` from its caller. Managers can share a
pool to enforce one idle-cache budget while keeping tensor keys and scheduling
independent.

- Tensors are keyed as `"{block_idx}_{tensor_idx}"`.
- Each manager owns its `GetCnt` tracker and weak references to live `SwapTensor` objects.
- Prefetch launches H2D for the previous block in the same sequence.
- `reset_async_activation_offload(model)` resets all managers attached to that root
  after a completed or failed step; the trainer invokes it at step boundaries.
  Callers with several model roots must reset each root.

### Per-Instance Class Patching

Python resolves `module(args)` through `type(module).__call__`. Each selected
module is therefore rebound to a private subclass that wraps `__call__`, leaving
its original class and unselected instances unchanged. Reapplying offload reuses
that subclass (guarded by `_veomni_async_offload_instance_class`).

Per-instance attributes set by `async_offload_modules`:
- `_veomni_offload_layer_idx`: Index in the offload sequence
- `_veomni_offload_depth`: Total number of offloaded modules
- `_veomni_offload_hidden_states_idx`: Position of hidden_states in args
- `_veomni_offload_prefetch`: Whether to enable prefetch
- `_veomni_offload_manager`: Manager for this matched sequence

### Prefetch Mechanism

During the backward pass, when unpacking a tensor from layer `N`, the prefetch
mechanism launches H2D for all tensors of layer `N-1`. This overlaps the H2D transfer
with the backward computation of layer `N`, reducing the latency of waiting for
activations to be available on GPU.

## Limitations

- **Gradient checkpointing is recommended:** Set
  `model.accelerator.gradient_checkpointing.enable=True` to recompute intermediate activations.
  Without it, async offload still works but intermediates remain on the accelerator,
  reducing the memory savings.
- **Pinned CPU memory:** Requires sufficient CPU pinned memory to hold all offloaded
  activations. For very large models with many layers, this can be substantial.
- **Single-stream architecture:** Currently uses a single swap stream for both D2H and
  H2D. Bi-directional overlap (simultaneous D2H and H2D) is not yet supported.
- **VLM multiple forward passes:** VLM models may invoke the same visual-block
  sequence twice per training step (image forward + video forward / FSDP dummy_forward).
  `GetCnt` handles this by incrementing existing counts instead of resetting, ensuring
  unique keys across passes.
