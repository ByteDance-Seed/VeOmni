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
[How a module's arguments are resolved](../usage/training_and_inference.md#how-a-modules-arguments-are-resolved))
so the entire graph runs at one uniform SP degree.

With SP size `S` on `W` ranks the data-parallel size is `W / S`. On a small
machine this can reach `dp = 1`; compensate with gradient accumulation (a larger
`global_batch_size` over the same `micro_batch_size`).

## 2. Why one SP size for the whole graph

`OmniTrainer` builds a single dataloader under the job's `base` parallel state
(`model.accelerator`), yielding `W / S` distinct shards and replicating each
shard across a base SP group of `S` ranks. Because each module's `pre_forward`
slices its replicated input and `post_forward` all-gathers across its SP group,
and because the gradient, loss, and metric reductions in section 4 pair each
module's SP group with its data-parallel group, every module inherits the same
`ulysses_size = S` from `model.accelerator` so its SP groups match the
dataloader's replication groups.

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
  SP group, so every rank in an SP group holds the **same** samples.
- **Slice in `pre_forward`.** A module with SP support branches on
  `get_parallel_state().sp_size > 1` and slices its replicated input to `1/S`
  with `sp_pad` + `slice_input_tensor` or `sp_pad_and_slice`. Sequence modules
  slice tokens; per-item encoders (SigLIP, VQVAE) slice along the batch of
  images.
- **Gather in `post_forward`.** The matching branch all-gathers with
  `gather_outputs` and strips the padding, so everything downstream of
  `post_forward` (writing hidden states back onto the conversation list,
  label alignment, loss) is SP-agnostic. Because every rank in the group starts
  from the same replicated sample, per-rank chunk shapes match for
  `gather_outputs`.
- **Encoder-to-LLM boundary.** Encoder outputs are gathered back to the full
  sequence, and the backbone assembles its packed sequence from full-length
  embeddings before slicing again for its own forward.

Forward and backward activation peaks are both about `1/S` of a full sample.
The node runner is
`veomni/models/seed_omni/accelerated/utils/executor.py::execute_train_node`, and
the primitives are in `veomni/distributed/sequence_parallel/data.py`.

## 4. Correctness invariants

**Gradients.** `gather_outputs` uses an autograd-aware gather: its backward
all-reduces (sum) over the SP group and keeps the local shard. That introduces a
factor of `S`, which FSDP2's gradient average over `dp_shard_sp` cancels,
because parameters are sharded and reduced on a mesh that includes SP. The
result matches the non-SP baseline; this is the same invariant as single-model
Ulysses in VeOmni.

**Loss requirement.** Decode-side losses must reduce with
`reduce_sequence_parallel_loss(..., group=ps.fsdp_group)`, a token-weighted
reduction over `dp_sp`. With replicated data the SP peers hold identical
`(ce_sum, n_valid)` pairs, so each distinct DP shard counts once and the SP
copies cancel. The current in-tree stand-in modules (`fake_model`) do not yet
apply this reduction; production decode modules (`text_encoder.decode`,
`vqvae.decode`) implement it.

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

## 5. Slicing dimensions

Sequence-parallel modules with Ulysses attention (text backbones, Qwen3-VL ViT)
slice along the flat token or patch sequence so even a single long item splits
evenly across `S` ranks. Block-diagonal image encoders whose attention operates
per image without Ulysses (SigLIP, VQVAE) slice the replicated image batch
across the `S` ranks instead, so each image is computed once on one rank and
gathered in `post_forward`.

## 6. Limitations and future work

- **Local operators in audio and video encoders.** Whisper-style conv1d
  downsampling and video 3D-conv patchify or temporal windows act on the
  sequence before attention, so encoders with cross-shard local operators need a
  whole-item scatter/gather path (not yet implemented).
- **Cross-sample compute balance.** Length- or compute-aware packing across DP
  groups is not yet implemented.
- **Per-module SP sizes.** All modules share one SP size (`model.accelerator.ulysses_size`).
