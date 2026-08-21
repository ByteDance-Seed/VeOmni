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
  - [OffloadManager Singleton](#offloadmanager-singleton)
  - [Class-Level __call__ Patching](#class-level-__call__-patching)
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
`train.accelerator.offload_config.enable_async_activation_offload` parameter in your
configuration file:

```yaml
train:
  gradient_checkpointing:
    enable: true              # Required: must be enabled together
  accelerator:
    offload_config:
      enable_async_activation_offload: true
```

Or via command-line:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3-moe.yaml \
    --train.gradient_checkpointing.enable true \
    --train.accelerator.offload_config.enable_async_activation_offload true
```

**Requirement:** `gradient_checkpointing.enable` must be `True` when using async
activation offload. The feature relies on recomputation to avoid storing intermediate
activations on GPU. If gradient checkpointing is disabled, a `ValueError` will be
raised at config validation time.

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

This combination yields zero persistent GPU memory for both hidden states (offloaded to
CPU) and intermediate activations (recomputed).

### Stream-Based D2H/H2D Overlap

Unlike the synchronous `custom_save_on_cpu` (which blocks the default stream until
the transfer completes), async activation offload uses a dedicated swap stream:

```
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

The `saved_tensors_hooks` stack ordering is critical. Class-level `__call__` patching
places `async_save_on_cpu` **outside** the `GradientCheckpointingLayer` checkpoint
boundary:

```
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

### Automatic Module Discovery

Offload targets are automatically derived from `model._no_split_modules` — the same
class names used for FSDP sharding boundaries. This mirrors the FSDP enablement pattern
where `_no_split_modules` determines sharding units automatically.

For example, with Qwen3-MoE where `_no_split_modules = ["Qwen3MoeDecoderLayer"]`,
all `Qwen3MoeDecoderLayer` instances are automatically identified as offload targets.

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train.accelerator.offload_config.enable_async_activation_offload` | bool | `False` | Enable async activation offload |
| `train.accelerator.offload_config.enable_activation` | bool | `False` | Enable synchronous activation offload (legacy) |
| `train.accelerator.offload_config.activation_gpu_limit` | float | `0.0` | GPU memory limit (GB) for synchronous offload |
| `train.gradient_checkpointing.enable` | bool | `True` | Must be `True` when using async offload |

When `enable_async_activation_offload` is `True`, it takes precedence over
`enable_activation` (synchronous offload). The global forward/backward contexts
become `nullcontext` since offloading is handled per-module via `saved_tensors_hooks`.

## Core API

### `apply_async_activation_offload(model)`

Apply async activation offload to matched submodules. Called automatically by
`build_parallelize_model` when `enable_async_activation_offload=True`.

```python
from veomni.distributed.async_offloading import apply_async_activation_offload

apply_async_activation_offload(model)
```

Args:
- `model`: The model to apply offloading to. Must have `_no_split_modules` attribute.

### `async_save_on_cpu(block_idx, depth, custom_check_fn, prefetch)`

A `saved_tensors_hooks` context that intercepts saved tensors and offloads them to
CPU. Applied per-module via class-level `__call__` patching.

```python
from veomni.distributed.async_offloading import async_save_on_cpu

with async_save_on_cpu(block_idx=0, depth=12, prefetch=True):
    output = module(*args, **kwargs)
```

Args:
- `block_idx`: Index of the current module in the offload sequence.
- `depth`: Total number of modules being offloaded.
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

### OffloadManager Singleton

`OffloadManager` is a singleton (via `_Singleton` metaclass) that manages all
`SwapTensor` instances. It provides:

- Key-based storage: tensors are keyed as `"{block_idx}_{tensor_idx}"`
- Prefetch coordination: `prefetch_get` launches H2D for the previous layer's tensors
- Reference counting via `OffloadItem`
- Global `GetCnt` tracker for consistent key assignment across forward passes

### Class-Level __call__ Patching

Module `__call__` is patched at the **class level** (not instance level) because
Python's dunder-method dispatch bypasses instance-level assignments: `module(args)`
always resolves to `type(module).__call__`.

Each unique class is patched only once (guarded by `_veomni_async_offload_patched`).
Instances not in the offload plan (lacking `_veomni_offload_layer_idx`) fall through
to the original `__call__` transparently.

Per-instance attributes set by `async_offload_modules`:
- `_veomni_offload_layer_idx`: Index in the offload sequence
- `_veomni_offload_depth`: Total number of offloaded modules
- `_veomni_offload_hidden_states_idx`: Position of hidden_states in args
- `_veomni_offload_prefetch`: Whether to enable prefetch

### Prefetch Mechanism

During the backward pass, when unpacking a tensor from layer `N`, the prefetch
mechanism launches H2D for all tensors of layer `N-1`. This overlaps the H2D transfer
with the backward computation of layer `N`, reducing the latency of waiting for
activations to be available on GPU.

## Limitations

- **Requires gradient checkpointing:** Async activation offload must be used together
  with gradient checkpointing (`train.gradient_checkpointing.enable=True`). Without
  recomputation, intermediate activations would consume GPU memory, negating the
  benefit of offloading hidden states.
- **Pinned CPU memory:** Requires sufficient CPU pinned memory to hold all offloaded
  activations. For very large models with many layers, this can be substantial.
- **Single-stream architecture:** Currently uses a single swap stream for both D2H and
  H2D. Bi-directional overlap (simultaneous D2H and H2D) is not yet supported.
- **VLM multiple forward passes:** VLM models may invoke the same visual-block
  sequence twice per training step (image forward + video forward / FSDP dummy_forward).
  `GetCnt` handles this by incrementing existing counts instead of resetting, ensuring
  unique keys across passes.
