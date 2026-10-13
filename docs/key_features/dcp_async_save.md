# Asynchronous DCP Checkpoint Saving

## Background

Large distributed checkpoints can pause training while model and optimizer
states are written to storage. VeOmni can submit the DCP tensor save through
PyTorch's asynchronous API and retain a future to wait for completion later.
This can overlap part of checkpoint I/O with subsequent training. It does not
make every checkpoint-related operation asynchronous or guarantee a stall-free
save.

This page follows revision `b8a3edcc01dba6026c785fa087a9ca1a59c9c9e2`.
Use a working distributed training recipe and the repository's pinned hardware
dependencies; see [Ascend installation](../get_started/installation/install_ascend_x86.md)
and [trainer usage](../usage/trainer.md).

## Features

### Overlap of checkpoint writes and training

A checkpoint records training state so the job can resume later. A distributed
checkpoint (DCP) typically uses multiple training processes to save model and
optimizer shards together; each process is called a rank.

A synchronous save waits for the tensor write to finish on the saving path before
returning to the training loop. An asynchronous save gives the DCP tensor write
to background work. Once the call returns, training may continue, and a later
operation that needs a complete checkpoint waits for the result.

```text
Sync:  train → prepare state → wait for tensor write → continue training
Async: train → prepare and submit → continue training ─────→ wait when needed
                                   background write ──→ complete
```

This is a schematic timeline: preparing/staging state and writing some sidecar
files can still block, so the entire save call is not an immediate return. The
opportunity is to overlap **part of storage I/O with subsequent computation**.

### Completion tracking and synchronization

VeOmni retains a future object to track the background result. Think of it as a
task result that is not available yet: successful submission hands the work off;
successfully waiting for the result confirms completion of that background work.

`wait_for_pending_save()` calls `future.result()` to wait. A background failure is
raised here. The wait clears the recorded future and, on success, executes a
barrier so participating processes synchronize at this point. A `saved` log or
an existing directory therefore cannot replace a completion check.

There is only one pending-save slot. Suppose step 100 starts an async save and it
is still running at the next save boundary: the new async save waits for it,
rather than building an ever-growing queue. Loading, HF/LoRA export and normal
training shutdown also wait for an existing save.

### Checkpoint state composition

| State | Examples | Current saving path |
| --- | --- | --- |
| Model and optimizer state | Parameters and supplied optimizer state | DCP tensor saving; `save_async` primarily controls this write path. |
| Model-related sidecar state | `extra_state`, such as learning-rate scheduler state | Written to sidecar files before submitting the DCP tensor save. |
| Job-level state | Data-loader position and random-number generator state | Saved per rank by the separate `GlobalStateCallback` on its schedule. |

Model state is required; optimizer state is included when an optimizer is supplied.
Enabling `save_async` does not make all writes in the last two rows asynchronous.
Recovering training progress requires both tensor shards and associated state
files, not just the presence of weights.

### Save scheduling and process groups

`CheckpointCallback` decides when to save by step or epoch.
`ModelCheckpointManager` forwards `train.checkpoint.save_async` to the DCP
checkpointer, which calls `dcp.async_save` and manages its future.

The first async save creates a dedicated Gloo process group for coordination
between saving processes. Even when model training uses CUDA or Ascend, those
processes therefore still need a working Gloo communication environment.

Custom loops should follow the standard trainer by waiting for pending saves
before destroying process groups or treating a checkpoint as ready. Normal-end
waiting only covers paths that can run cleanup; it does not guarantee completion
when a process is forcibly terminated.

## Configuration

Fields are under `train.checkpoint`.

| Field | Default | Meaning |
| --- | --- | --- |
| `manager` | `dcp` | Select the DCP checkpointer. |
| `save_async` | `false` | Submit the DCP tensor write asynchronously. |
| `output_dir` | `output` | Run output root; DCP lives under its `checkpoints` subdirectory. |
| `save_steps` | `0` | DCP step interval; zero disables step scheduling. |
| `save_epochs` | `1` | DCP epoch interval; zero disables epoch scheduling. |
| `stage_dir` | `null` | Optional staging directory for the synchronous path; a nonempty value cannot be combined with `save_async`. |
| `dcp_save_to_lowest_rank` | `false` | Route replicated shards to their lowest holder rank instead of distributing replica writes. Does not consolidate unique EP/TP/PP shards. |
| `load_path` | `null` | Specific completed checkpoint directory to resume. |
| `save_hf_weights` | `true` | Enable HF exports; those use a separate path and are not made async by `save_async`. |
| `hf_save_steps` / `hf_save_epochs` | `0` / `0` | Periodic HF export intervals. An enabled final HF export can still occur. |

Step and epoch schedules are independent, with duplicate saves at the same step
suppressed by the callback. `save_async: true` alone does not enable a step
schedule. Ending training does not unconditionally create a new final DCP;
it drains the pending one and handles the separately configured final HF export.

## Usage example and recovery

Merge this **configuration fragment** into an existing distributed recipe:

```yaml
train:
  checkpoint:
    manager: dcp
    output_dir: output/async-check
    save_async: true
    save_steps: 100
    save_epochs: 0
    stage_dir: null
    dcp_save_to_lowest_rank: false
    save_hf_weights: false
```

Keep the recipe's model/data/optimizer settings and run its original launch
command. The directory for optimizer step 100 is:

```text
output/async-check/
  model_assets/
  checkpoints/
    global_step_100/
      ... DCP metadata, tensor shards and state sidecars ...
```

Do not set the derived `save_path` field in YAML. For recovery after successful
completion, merge this setting into an otherwise compatible recipe:

```yaml
train:
  checkpoint:
    load_path: output/async-check/checkpoints/global_step_100
```

The examples are verifiable configuration fragments, not a self-contained training
job. Validate on the target hardware with a short disposable run: use a small
save interval, run through two saves, allow normal cleanup to finish, and resume
from a completed step. Verify restored optimizer/scheduler state, global step and
data-loader position, then continue at least one optimizer step. For a controlled
baseline comparison, keep topology, data ordering and seeds fixed.
