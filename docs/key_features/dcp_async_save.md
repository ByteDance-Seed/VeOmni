# Asynchronous DCP Checkpoint Saving

## Background

Large distributed checkpoints can pause training while model and optimizer
states are written to storage. VeOmni can submit the DCP tensor save through
PyTorch's asynchronous API and retain a future to wait for completion later.
This can overlap part of checkpoint I/O with subsequent training. It does not
make every checkpoint-related operation asynchronous or guarantee a stall-free
save.

The repository's pinned hardware
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

### Independent save slots and completion

VeOmni tracks weights and optimizer writes separately. Each kind has one future
(the handle used to retrieve its background result) and its own Gloo process
group. The weights slot is named `ckpt`; the optimizer slot is named `optimizer`.
The two writes can overlap, although state preparation and submission still run
on the calling path.

A new submission waits only for the previous write **in the same slot**. For
example, a weights save at step 200 drains the previous weights future before
reusing that slot; it does not directly drain the optimizer future. The optimizer
submission later drains its own previous future. This bounds the outstanding
work to one write per slot, rather than accumulating a queue. These slots belong
to the checkpointer class, not to each step or each model: a multi-model job also
reuses them across model submissions.

`wait_for_pending_save()` drains every recorded future. It first waits for all
local tasks, collecting failures even if one task raises, then performs failure
agreement on the corresponding slot's Gloo group. A reported save failure is
therefore propagated across ranks. This coordination is not the old single-future
wait followed by a default-group barrier.

Loading, export preparation and normal training shutdown use this completion
entry point. An early `saved` log or an existing directory does not mean that
both tensor writes have finished.

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
checkpointer, which calls `dcp.async_save` and manages the two slots' futures.

The standard manager also calls a training-group barrier after submitting the
saves. That barrier coordinates the calling ranks; it does not resolve the
background futures or establish that their writes have completed.

The first async submission for each slot creates that slot's dedicated Gloo
process group. Separate groups keep concurrent weights and optimizer collectives
from sharing one background communication sequence. Even when model training uses CUDA or Ascend, those
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
| `save_timeout_seconds` | `null` | Positive integer collective timeout for async-slot and staged-save Gloo groups; unset uses Gloo's default. Applied when each cached group is first created, not a per-file I/O deadline. |
| `dcp_save_to_lowest_rank` | `false` | Route replicated shards to their lowest holder rank instead of distributing replica writes. Does not consolidate unique EP/TP/PP shards. |
| `load_path` | `null` | Specific completed checkpoint directory to resume. |
| `save_hf_weights` | `true` | Enable HF exports; those use a separate path and are not made async by `save_async`. |
| `hf_save_steps` / `hf_save_epochs` | `0` / `0` | Periodic HF export intervals. An enabled final HF export can still occur. |

Step and epoch schedules are independent, with duplicate saves at the same step
suppressed by the callback. `save_async: true` alone does not enable a step
schedule. Ending training does not unconditionally create a new final DCP;
it drains pending writes. If final HF/LoRA export is enabled, export preparation
can first create a DCP for the final step when that step has not yet been saved,
and then wait for it. The example below disables that export.

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
      checkpoint_manifest.json
      model/
        ckpt/                 # .metadata + .distcp
        optimizer/            # .metadata + .distcp
        extra_state/extra_state_rank_0.pt # one file per rank
      loader/rank_0.pt        # one file per rank
      extra_state/rank_0.pt   # one file per rank
```

This shows a single-model job with optimizer state; `rank_0.pt` represents each
rank's file. Multi-model jobs nest each model's state under `model/<module>/`.
At this revision, scheduler state is inside the model's per-rank `extra_state`
dictionary, together with any runtime-specific state.

`checkpoint_manifest.json` covers trainer-level cursor/state completion only.
It can appear while DCP is still writing. For each discovered model, automatic
resume discovery requires `.metadata` in its weights directory and, **if its
optimizer directory exists**, in that directory too. A weights-only save therefore
does not require an optimizer marker. A manifest with `model_state: false`
requires no model markers. These are marker-existence checks, not validation of
every shard, sidecar or file's contents; successful discovery alone does not prove
that a full training resume will succeed. See [checkpoint layout](../usage/checkpoint.md) for discovery and
legacy layouts; the per-rank model sidecar shown above follows this revision's
`_save_extra_state` implementation.

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