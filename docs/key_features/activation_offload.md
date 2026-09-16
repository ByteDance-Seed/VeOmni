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

| Gradient checkpointing | Where `activation_gpu_limit` applies |
| --- | --- |
| Disabled | Controls retained activations when saving tensors during forward. |
| Enabled | Forward uses a zero retention threshold for inter-layer activations; the parameter applies to tensors saved during backward recomputation. |

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
| `activation_gpu_limit` | `0.0` | Synchronous retained-activation threshold, in units of 1,024³ bytes. Use a non-negative value. Does not tune async mode. |
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
      activation_gpu_limit: 1.0
      enable_async_activation: false
```

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