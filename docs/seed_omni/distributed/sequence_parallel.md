# SeedOmni Sequence Parallelism

SeedOmni runs classic single-pass Ulysses sequence parallelism (SP) under one
SP size shared by the whole graph. Each module that supports SP slices its input
inside its own `pre_forward`, runs one forward, and all-gathers the result in its
own `post_forward`. This page describes that contract and why it gives the same
gradients and metrics as a non-SP run. General Ulysses background is in
[Ulysses](../../key_features/ulysses.md).
Everything else a module may choose for itself (FSDP2, DDP, extra parallel
groups) is in [Per-Module Parallelism](per_module_parallelism.md).

## 1. Enabling SP

Set the SP size on the global accelerator:

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.accelerator.ulysses_size 4
```

Every module inherits `ulysses_size` through the accelerator deep-merge in
`build_omni_module_runtime_args` (see
[How a module's arguments are resolved](../usage/training_and_inference.md#how-a-modules-arguments-are-resolved)).
There is no dedicated SP config file. The framework does **not** validate that
all modules agree, so do not set a per-module `ulysses_size` in the `modules`
YAML; the next section explains why.

With SP size `S` on `W` ranks the data-parallel size is `W / S`. On a small
machine this can reach `dp = 1`; compensate with gradient accumulation (a larger
`global_batch_size` over the same `micro_batch_size`).

## 2. Why one SP size for the whole graph

Module boundaries do not depend on the SP size. Every module gathers its output
back to the full sequence in `post_forward`, and the next module slices its own
input again in `pre_forward` (section 3), so on paper each module could cut its
input into a different number of pieces. What ties the modules together is the
data they slice.

**One data stream, replicated once.** `OmniTrainer` builds a single dataloader
under the job's `base` parallel state, which comes from `model.accelerator`. It
yields `W / S` distinct shards and replicates each one across a base SP group of
`S` ranks. Slicing is only correct when every rank of a module's SP group holds
the same sample, that is, when the module's SP groups sit inside the groups the
loader replicated over. With a module SP size `S'`:

- **`S' = S`.** The module's groups are the replication groups. This is the
  supported case.
- **`S'` larger than `S`, or groups that do not nest.** The module's SP peers
  hold different samples. Each rank slices its own sample, and `gather_outputs`
  concatenates chunks of different samples: the all-gather fails or hangs when
  the chunk shapes differ, and when they happen to match it silently stitches
  unrelated samples into one sequence.
- **`S'` smaller than `S`, groups nested.** Shapes stay consistent, but `S / S'`
  ranks run the identical full computation on the same replicated input. The
  module saves no memory and wastes compute, which defeats the reason to lower
  its SP size.

**The reductions are derived for one factor.** The gradient, loss and metric
reductions in section 4 each pair an operation over the SP group with one over
the data-parallel ranks, and they are derived and tested for modules whose SP
group equals the replication group. No other combination is validated.

**Nothing enforces it.** `ulysses_size` reaches each module through the
accelerator deep-merge, so an override in the `modules` YAML is accepted
without an error. Set it only on `model.accelerator`.

Supporting different SP sizes per module would need the loader to replicate
over the largest SP group, every module's SP size to divide it so that the
groups nest, and the section 4 reductions to be re-derived for the smaller
groups. None of this is implemented.

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
there is no separate SP hook. Forward and backward activation peaks are both
about `1/S` of a full sample. The node runner is
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

The hard rules agents must follow are kept in `.agents/knowledge/constraints.md`
(§7).

## 5. Choosing between slicing and item balancing

For a **block-diagonal** encoder (each image or clip attends only within itself)
whose slice boundaries align with items, SP slicing is exactly data balancing:
each item is computed once on one rank and no all-to-all is needed. SP slicing
also balances load strictly better than assigning whole items to ranks, because
it splits by tokens: a single image is still split `S` ways, with at most `S`
padding tokens once per batch, while whole-item assignment idles ranks when there
are fewer items than ranks or the items differ in size. For SP-aware encoders
(global attention, no cross-shard locality, such as a ViT), uniform SP slicing is
therefore sufficient, and it is what every module does today.

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
- **Per-module SP sizes.** All modules share one SP size; there is no supported
  way to run, for example, the backbone at SP 4 and an encoder at SP 1. See
  [section 2](#2-why-one-sp-size-for-the-whole-graph) for what it would take.
