# Activation Offload

## Background

Activations saved for backward can dominate accelerator memory, especially with long
sequences. Activation offload stores selected saved tensors on the host and restores
them when autograd needs them. It trades host memory and transfer time for accelerator
capacity. It does not offload model parameters or optimizer states, and does not
guarantee a throughput improvement.

Follow the
[installation guide](../get_started/installation/install.md), or the
[Ascend x86 guide](../get_started/installation/install_ascend_x86.md) /
[Ascend ARM guide](../get_started/installation/install_ascend_arm.md), before using a
working model and dataset recipe. The YAML below contains **configuration fragments**;
merge the fields into that recipe, preserving its other `model` and `train` settings.
Do not launch these fragments as complete training configurations.

## Features

### 1. basic mechanism

Each layer produces intermediate results during the forward pass. Some must be
saved so that backward can compute gradients; these are the activations discussed
here. They can still occupy device memory while later layers are running.

Activation offload uses this waiting period: **move a saved activation to CPU
memory while it is not needed, then bring it back when backward needs it**.
For an eligible layer input, the flow is:

```text
Forward: save layer input → move it to CPU → compute later layers
Backward: need that input → restore it to the device → compute gradients
```

Fewer activations may remain on the device at once, at the cost of host memory
and transfers. The benefit depends on how much can be offloaded and whether
transfers finish in time.

### 2. Synchronous vs. Asynchronous Offloading

| Mode | How tensors move | Effect on training |
| --- | --- | --- |
| Synchronous | Transfers occur on the save/restore execution path. | Transfer waits directly enter that path; the mechanism is straightforward. |
| Asynchronous | A separate transfer stream moves tensors and attempts to prefetch upcoming tensors during backward. | Transfers may overlap computation, but a tensor must still be ready before it is used. |

Asynchronous does not mean using data before its transfer completes, nor does it
guarantee that all transfer time is hidden. Select one mode; they are mutually
exclusive.

### 3. Relationship with Gradient Checkpointing

Gradient checkpointing saves fewer intermediate results and recomputes them during
backward. Offload changes where saved results live. They can work together: save
and offload a layer input, restore it during backward, then recompute the required
intermediate results from that input.

The current async implementation primarily targets the input `hidden_states` of
selected modules. With gradient checkpointing, inputs can be offloaded while
intermediate activations are recomputed. Without checkpointing, intermediate
activations generally remain on the device. The last selected block retains its
saved tensor on-device. Enabling async offload therefore does not offload every
tensor produced by a layer.

### 4. Implementation details and retention thresholds

PyTorch saved-tensor hooks allow custom logic when saving and retrieving tensors
needed by backward. VeOmni uses them to implement the transfers above.

The synchronous `custom_save_on_cpu` skips tensors of at most 1,024 bytes and
weight views identified by the `TBackward0` heuristic. Other eligible tensors stay
on-device until the retention counter reaches the threshold; subsequent tensors
move to CPU. The trainer uses pageable host memory (`pin_memory=False`). The
synchronous threshold applies as follows:

| Gradient checkpointing             | Where `activation_gpu_limit` applies                                                                                                                                                                                                                                                    |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Disabled                           | Controls retained activations when saving tensors during forward.                                                                                                                                                                                                                       |
| Enabled, `enable_reentrant: true`  | The outer forward offload context uses a zero threshold. During backward recomputation, the outer backward offload hook can apply `activation_gpu_limit` to eligible saved tensors.                                                                                                     |
| Enabled, `enable_reentrant: false` | The outer forward offload context uses a zero threshold for eligible checkpoint inputs. PyTorch's inner recomputation hook takes precedence over the outer backward offload hook, so `activation_gpu_limit` does **not** control retained activations inside checkpoint recomputation.  |

The distinction comes from hook nesting: only the innermost saved-tensor hook
handles a tensor. Non-reentrant checkpointing installs its own hooks to recover
intermediate results by recomputation. An outer offload context does not override
that behavior. Tensors saved outside those checkpoint regions can still reach
the outer hook; this is not a claim that all offloading is disabled.

This is a saved-tensor accounting threshold, **not a limit on total device memory**;
retaining a whole tensor can also cross it.

Async wrapping is applied before FSDP2 sharding. With gradient checkpointing, the
offload hook sits outside the checkpoint boundary to capture saved inputs. When
`hidden_states` is identifiable, its data pointer selects target tensors; matching
views may be copied into private contiguous storage before handling them.

## Configuration

All fields below are under `model.accelerator.offload_config`.

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_activation` | `false` | Enable synchronous offload. |
| `activation_gpu_limit` | `0.0` | Synchronous retained-activation threshold, in units of 1,024³ bytes. Use a non-negative value; this field is not validated. A negative value bypasses retention for eligible tensors (the small-tensor and weight-view exclusions still apply). Does not tune async mode. |
| `enable_async_activation` | `false` | Enable asynchronous offload; cannot be combined with `enable_activation`. |
| `activation_offload_modules` | `[]` | Async module paths. Empty means discover module classes from `model._no_split_modules`. Missing metadata or no matches raises an error. |
| `activation_offload_host_cache_limit_gb` | `4.0` | Async idle host-buffer cache cap per pool, in units of 1,024³ bytes. Must be finite and non-negative and fit the implementation's byte conversion bound. Zero disables reuse. |

Module patterns are segment-aware: `model.layers.*` matches direct children only,
not `model.layers.0.self_attn`. The sequential-group syntax `model.layers.{*}` is
also supported. Paths must match the actual model instance; a multimodal model
may use a different prefix. Inspect `model.named_modules()` before overriding
auto-discovery. Matching modules alone does not prove that their saved inputs
will be eligible for offload.

## Usage examples

Start with synchronous offload to establish correctness:

```yaml
model:
  accelerator:
    gradient_checkpointing:
      enable: true
      enable_reentrant: false
    offload_config:
      enable_activation: true
      activation_gpu_limit: 0.0
      enable_async_activation: false
```

This example uses non-reentrant checkpointing. The explicit `0.0` is the default;
increasing it does not tune retention inside checkpoint recomputation. Choose
checkpoint mode according to model compatibility, not just to make this knob active.

For asynchronous offload, replace the preceding offload settings with:

```yaml
model:
  accelerator:
    gradient_checkpointing:
      enable: true
      enable_reentrant: false
    offload_config:
      enable_activation: false
      enable_async_activation: true
      activation_offload_modules: []
      activation_offload_host_cache_limit_gb: 4.0
```

Use the original recipe's launch command after merging. Confirm the startup log
contains `Applying activation offload to module` for async mode. Compare a short
baseline and offloaded run with identical inputs, seed, batch size and optimizer:
check finite loss/gradients, peak accelerator memory, peak host memory and steady
step time. Measure after warmup; do not infer a memory saving from the flag alone.

## Asynchronous offload internals

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
  avoiding persistent storage of those intermediate activations between forward and recomputation.

This ordering requires checkpointing to occur inside the selected module's
`__call__`, as in `GradientCheckpointingLayer`. Wan instead invokes checkpointing
outside the block call. Enabling async offload there places the offload hooks
inside the checkpoint hooks and suppresses recomputation. The Wan I2V examples
therefore keep gradient checkpointing enabled and async offload disabled; an
integration around the outer checkpoint call is needed before combining them.

### Automatic Module Discovery

Offload targets are automatically derived from `model._no_split_modules` — the same
class names used for FSDP sharding boundaries. This mirrors the FSDP enablement pattern
where `_no_split_modules` determines sharding units automatically.
For example, with Qwen3-MoE where `_no_split_modules = ["Qwen3MoeDecoderLayer"]`,
all `Qwen3MoeDecoderLayer` instances are automatically identified as offload targets.


### Core API

#### `apply_async_activation_offload(model, activation_offload_modules, ...)`

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
  targets from `model._no_split_modules`. Missing or unmatched targets raise an error.
- `host_cache_limit_bytes`: Optional idle pinned-buffer cache budget for this call.
- `host_buffer_pool`: Optional shared `PinnedBufferPool` for multiple model roots;
  mutually exclusive with `host_cache_limit_bytes`.

#### `async_save_on_cpu(block_idx, depth, manager, custom_check_fn=None, prefetch=True)`

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

### Implementation Details

#### SwapTensor Lifecycle

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

#### OffloadManager Ownership

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

#### Per-Instance Class Patching

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

#### Prefetch Mechanism

During the backward pass, when unpacking a tensor from layer `N`, the prefetch
mechanism launches H2D for all tensors of layer `N-1`. This overlaps the H2D transfer
with the backward computation of layer `N`, reducing the latency of waiting for
activations to be available on GPU.

### Limitations

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
