# SeedOmni Sequence & Context Parallelism

SeedOmni runs single-pass sequence/context parallelism under one shared
`dp_size` (and therefore one unified `sp_size = cp_size * ulysses_size`) across
all distributed modules in the graph. Each module may independently configure
Ulysses sequence parallelism (`ulysses_size`), context parallelism (`cp_size`),
or hybrid `cp_size * ulysses_size`, as long as every distributed module resolves
to the same `dp_size`. Inside its own `pre_forward`, each module slices its
replicated input according to its own parallel state, runs one forward, and
all-gathers the result in its own `post_forward`. This page describes that
contract and why it gives the same gradients and metrics as a non-SP run.
General Ulysses background is in [Ulysses](../../key_features/ulysses.md).
Everything else a module may choose for itself (FSDP2, DDP, extra parallel
groups) is in [Per-Module Parallelism](per_module_parallelism.md).

## 1. Enabling SP / CP

Set a uniform `ulysses_size` on the global accelerator:

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/fake_model/train/base.yaml \
  --model.accelerator.ulysses_size 4
```

Or, keeping `--model.accelerator.ulysses_size 4` (or `model.accelerator.ulysses_size: 4`
in `base.yaml`) so the top-level accelerator also resolves to `dp_size = W / 4`,
configure `ulysses_size` / `cp_size` per module in `modules_train.yaml` (for
example, Ulysses SP on the vision encoder and Context Parallelism on a
DeepSeek-V4 LLM backbone), provided every distributed module and
`model.accelerator` resolve to the same
`dp_size = world_size // (cp_size * ulysses_size)`:

```yaml
vision_encoder:
  accelerator:
    ulysses_size: 4
    cp_size: 1

llm_backbone:
  accelerator:
    ulysses_size: 1
    cp_size: 4
```

Every module inherits from `model.accelerator` through `_deep_update` in
`build_omni_module_runtime_args` (see
[How a module's arguments are resolved](../usage/training_and_inference.md#how-a-modules-arguments-are-resolved)).
Because `_deep_update` merges keys individually, if `model.accelerator` sets
`ulysses_size > 1` and a module wants pure CP (`cp_size > 1`), remember to also
set `ulysses_size: 1` in that module's `accelerator:` block so the two axes do
not multiply into a different `dp_size`.

`build_omni_model_runtime_args` validates that all distributed modules and the
top-level `model.accelerator` resolve to the same `dp_size`.

With `sp_size = cp_size * ulysses_size = S` on `W` ranks the data-parallel size
is `W / S`. On a small machine this can reach `dp = 1`; compensate with gradient
accumulation (a larger `global_batch_size` over the same `micro_batch_size`).

## 2. Why all modules share one `dp_size`

Module boundaries do not depend on whether a module uses Ulysses SP, CP, or
`cp * sp`. Every module gathers its output back to the full sequence in
`post_forward`, and the next module slices its own input again in `pre_forward`
(section 3). What ties the modules together is the `dp_size` and the replication
group over which the dataloader distributes identical samples.

**DeviceMesh layout.** `ParallelState` always constructs its device mesh in the
order `[pp, dp_replicate, dp_shard, cp, ulysses, tp]`. When `pp_size = 1` and
`tp_size = 1`, any two modules with the same `dp_size` have the same unified
`sp_size = cp_size * ulysses_size = W / dp_size`, the exact same `dp_rank`
(`global_rank // sp_size`), the exact same `dp_group`, the exact same contiguous
replication group `sp_group` (`[dp_rank * sp_size, (dp_rank + 1) * sp_size)`),
and the exact same `fsdp_group` (`dp_sp`) — regardless of how each module
factors `sp_size` into `cp_size * ulysses_size` or splits `dp_size` into
`dp_replicate * dp_shard`.

**One data stream, replicated once per `dp_rank`.** `OmniTrainer` builds a
single dataloader under the job's `base` parallel state using the shared
`dp_size`. It yields `dp_size = W / S` distinct shards and replicates each one
across the `S` ranks of its `sp_group`. Each module knows its own `ParallelState`
(`dp_size`, `ulysses_size`, `cp_size`, `sp_size`) and decides inside its own
scoped execution how to process the replicated data.

**Enforced at argument resolution.** `build_omni_model_runtime_args` checks that
every distributed module resolves to the same `dp_size` and raises `ValueError`
before any model weights or process groups are built if any module disagrees.

## 3. Data flow

```text
# dataloader: W/S distinct shards; each shard is replicated to every rank of its SP group
for node in training_graph:
    kw  = module.pre_forward(**data)    # sp_size > 1: pad + slice 1/S (sequence or batch dim)
    out = module(**kw)                  # one forward; attention does the in-group all-to-all
    out = module.post_forward(**out)    # sp_size > 1: gather_outputs back to the full sequence, unpad
# one loss.backward() over the whole graph; FSDP2 reduces gradients on a mesh that includes SP
```

- **Replicated data.** The standard `BaseTrainer` sharded loader gives
  `dp_size = W / S` distinct shards and replicates each one to all ranks of its
  SP group, so every rank in an SP group holds the **same** samples. The
  collator does no modality-specific splitting.
- **Slice in `pre_forward`.** A module with SP support branches on
  `get_parallel_state().sp_size > 1` and slices its replicated input to `1/S`
  with `sp_pad` + `slice_input_tensor` or `sp_pad_and_slice`. Sequence modules
  slice tokens; per-item encoders (SigLIP, VQVAE) slice along the batch of
  images.
- **Gather in `post_forward`.** The matching branch all-gathers with
  `gather_outputs` and strips the padding, so everything downstream of
  `post_forward` (writing hidden states back onto the conversation list,
  label alignment, loss) is SP-agnostic. `gather_outputs` concatenates
  per-rank tensors, so every rank of the group must produce the same shapes,
  which replication guarantees.
- **Encoder-to-LLM boundary.** Encoder outputs are gathered back to the full
  sequence, the backbone assembles its packed sequence from full-length
  embeddings, then slices again for its own forward. There is no extra reshard
  mechanism between modules.

The SP logic lives entirely in each module's `pre_forward` / `post_forward`;
there is no separate SP hook. Inside the module's forward and backward pass,
local intermediate activations scale as about `1/S` of a full sample, while
`post_forward`'s `gather_outputs` materializes the full-sequence module output
on each SP rank before stripping padding. The node runner is
`veomni/models/seed_omni/accelerated/utils/executor.py::execute_train_node`, and
the primitives are in `veomni/distributed/sequence_parallel/data.py`.

## 4. Correctness invariants

**Gradients.** `gather_outputs` uses an autograd-aware gather: its backward
all-reduces (sum) over the SP group and keeps the local shard. That introduces a
factor of `S`, which FSDP2's gradient average over `dp_shard_sp` cancels,
because parameters are sharded and reduced on a mesh that includes SP. The
result matches the non-SP baseline; this is the same invariant as single-model
Ulysses in VeOmni.

**Loss.** Decode-side losses reduce with
`reduce_sequence_parallel_loss(..., group=ps.fsdp_group)`, a token-weighted
reduction over `dp_sp`. With replicated data the SP peers hold identical
`(ce_sum, n_valid)` pairs, so each distinct DP shard counts once and the SP
copies cancel.

**Metrics.** `metric_meter_set_seqlens` runs in `pre_forward` **before** the
slice and records full sample lengths. The step meter reduces over `dp_group`,
which excludes SP, so replicated SP ranks are not double-counted and the reported
tokens, FLOPs and MFU are identical across SP sizes. See
[Metric Meter](../mixins/metric_meter.md).

**DDP modules.** A module wrapped in DDP must still reduce over `dp_sp`, not
`dp` alone, even when its forward never slices, because its inputs come from SP
peers whose loss shares already account for the replication.

The hard rules agents must follow are kept in
`.agents/knowledge/veomni-seed-omni.md` (§3) and
`.agents/knowledge/constraints.md` (§7).

## 5. Choosing between slicing and item balancing

There are two ways to distribute encoder work across the `S` ranks of an
`sp_group`:

- **Item-aligned (batch-dim) slicing.** For a **block-diagonal** encoder where
  each image or clip attends only within itself (such as SigLIP or VQVAE without
  Ulysses), slicing along the batch of items assigns whole items to ranks and
  needs no cross-rank communication inside the encoder. However, a single item
  cannot be split across `S` ranks without communication, so ranks can idle when
  there are fewer items than `S` or when item sizes vary.
- **Token-level sequence slicing.** For **SP-aware** encoders whose attention
  exchanges heads/tokens across the SP group (such as a ViT with Ulysses
  attention), slicing the flat token or patch sequence splits even a single
  image `S` ways (with at most `S - 1` padding tokens per batch) and balances
  compute across all `S` ranks, at the cost of the in-attention all-to-all. Encoders
  with local pre-attention operators (section 6) cannot split a single item by
  tokens without halo exchange.

## 6. Limitations and future work

- **Local operators in audio and video encoders.** Whisper-style conv1d
  downsampling and video 3D-conv patchify or temporal windows act on the sharded
  sequence **before** any all-to-all, so shard boundaries would need halo
  exchange that Ulysses does not provide. Such encoders need an item-balancing
  path (keep each item whole on one rank, scatter items and gather embeddings)
  instead of SP slicing. This path is not implemented.
- **Cross-sample compute balance.** A text-only sample and a long video sample
  differ by orders of magnitude in compute, so DP groups can straggle.
  Length- or compute-aware packing at the dataloader level is not implemented.
- **Mismatched `dp_size` across modules.** All distributed modules share one
  `dp_size` (and therefore one `sp_size = cp_size * ulysses_size`); there is no
  supported way for different modules to use different `dp_size` values. See
  [section 2](#2-why-all-modules-share-one-dp_size).
