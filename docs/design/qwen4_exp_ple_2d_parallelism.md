# Qwen4-Exp PLE Two-Dimensional Parallelism

## Status

**Implemented** | 2026-09-01

This document specifies a Qwen4-Exp-only parallel layout for the PLE n-gram
embedding tables. It is not a proposal for general tensor parallelism in
VeOmni. The global `tp_size` remains `1`.

The design keeps every PLE weight permanently sharded over two axes and routes
only lookup requests and selected embedding vectors. It never all-gathers a
PLE vocabulary partition during forward or backward.

## Motivation

The released PLE weights are approximately 95 GiB. The current implementation
uses the existing ExtraParallel/FSDP2 composition:

1. `ple_size` shards each checkpoint-native table on dimension 0.
2. The complementary `ple_fsdp` mesh shards each local row partition on
   dimension 1 for storage.
3. Before PLE forward, FSDP2 all-gathers dimension 1 and materializes the full
   embedding width for this rank's row partition.

For a table with shape `[V, E]`, with `P = ple_size` and
`F = ple_fsdp_size`, the persistent and forward layouts are:

```text
full table:          [V,     E]
persistent/rank:     [V / P, E / F]
current forward:     [V / P, E]
```

The persistent state is well sharded, but the forward peak is still `M / P`
bytes for a table containing `M` bytes. FSDP2 may perform the same parameter
all-gather again in backward when the module reshards after forward.

The existing row-parallel lookup does **not** all-gather the complete
vocabulary. It routes IDs to row owners and returns selected vectors with
all-to-all. The problem is specifically the FSDP2 parameter all-gather over
`ple_fsdp`.

Simply swapping the axes -- `ple: Shard(1)` and
`ple_fsdp: Shard(0)` -- does not solve the problem. It changes the forward
layout to `[V, E / P]`, whose size is still `M / P`, and it requires all ranks
in the `ple` group to process identical token IDs. VeOmni currently assigns
different data-parallel samples to those ranks.

## Goals

- Keep each PLE table permanently sharded as `[V / P, E / F]`.
- Eliminate PLE parameter all-gather from forward and backward.
- Preserve different input samples on different data-parallel ranks.
- Load only the local two-dimensional PLE rectangle from the released
  checkpoint.
- Preserve differentiable lookup, global gradient averaging, gradient
  clipping, AdamW state, and DCP save/resume semantics.
- Allow PLE to coexist with MoE expert parallelism while keeping the PLE and
  expert parameter sets on their own ExtraParallel meshes.
- Keep the change specific to Qwen4-Exp PLE; do not enable or expose general
  model tensor parallelism.
- Leave the PLE output shape and the downstream `key_proj`, `value_proj`, norm,
  and convolution code unchanged.

## Non-goals

- General `tp_size > 1` support.
- Tensor-parallel attention, MLP, PLE projections, or LM head.
- PLE with Ulysses or context parallelism.
- HSDP replicas of the two-dimensional PLE table in the first implementation.
- FSDP CPU offload for FSDP-ignored PLE parameters.
- Hugging Face export of the complete 95 GiB PLE table.

Unsupported combinations must fail during argument or parallel-plan
validation. They must not silently fall back to the current parameter
all-gather path.

## Terminology and topology

The existing PLE ExtraParallel mesh is retained:

```text
mesh dimensions: [ple_fsdp, ple]
mesh shape:      [F, P]
```

The names retain backward compatibility, but their PLE-specific runtime roles
become:

| Mesh dimension | PLE role | Tensor placement |
|----------------|----------|------------------|
| `ple` | vocabulary/row owner | `Shard(0)` |
| `ple_fsdp` | embedding-column owner | `Shard(1)` |

`ple_fsdp` remains an FSDP dimension for ordinary model parameters. It acts as
a persistent column-sharding dimension only for PLE embedding parameters.
Code in the Qwen4-Exp path should use local aliases such as `ple_row_*` and
`ple_col_*` to make this distinction explicit.

For the initial implementation:

```text
P * F == world_size
P > 1
tp_size == 1
pp_size == 1
dp_replicate_size == 1
ulysses_size == 1
cp_size == 1
ep_size divides world_size
```

Each rank has a coordinate `(col_rank, row_rank)` in the `[F, P]` mesh.
Code must derive group-local ranks and destination ranks from `DeviceMesh`; it
must not assume that global ranks always equal `col_rank * P + row_rank`.

When EP is enabled, it owns a separate `[ep_fsdp, ep]` view of the same data
parallel ranks. Qwen4-Exp expert weights are sharded on expert dimension 0 over
`ep` and then on hidden dimension 1 over `ep_fsdp`. PLE and expert parameter
patterns must be disjoint; `ParallelPlan.apply()` rejects a parameter that
matches more than one enabled ExtraParallel dimension.

## Comparison with column-only PLE TP

An alternative is a 32-way PLE-only column parallel layout with no vocabulary
row sharding:

```text
weight/rank:       [V, E / 32]
input IDs:         identical on all 32 ranks
local lookup:      [K, E / 32]
output collective: all-gather -> [K, E]
```

This is conventional tensor-parallel embedding. It is simpler than the
two-dimensional lookup and its regular all-gather is generally easier to
optimize than variable-size all-to-all. It also avoids all-gathering table
weights. However, it is not equivalent to the proposed design under VeOmni's
current execution model.

| Property | Column-only TP=32 | PLE row × column, `P * F = 32` |
|----------|-------------------|---------------------------------|
| PLE parameter bytes/rank | `M / 32` | `M / 32` |
| PLE parameter all-gather | None | None |
| Input IDs inside the group | Must be identical | May differ on every rank |
| Unique data-parallel batches/group | 1 | 32 |
| Embedding result communication | Regular all-gather | Variable-size request/result all-to-all |
| Request metadata | None | Approximately `O(K * F)` integers |
| Load balancing | Naturally balanced | Depends on hashed-row distribution |
| Downstream dense compute | Repeated on all 32 ranks | Different data on every rank |
| Divisibility requirement | `D % 32 == 0` | `V % P == 0`, `D % F == 0` |

The two layouts have the same PLE parameter memory when they use the same 32
ranks. Their main difference is training throughput, not parameter capacity.

With PLE-only TP=32, all ranks must receive the same input IDs so that column
slice `i` corresponds to the same requested rows on every rank. Because the
rest of the model is not tensor parallel, all 32 ranks then repeat the same
dense-model forward and backward computation. The effective data-parallel
degree becomes `world_size / 32`; it is `1` for a 32-rank job. Matching the
unique-token batch of 32 independent data ranks requires either a 32-times
larger local batch or 32-times more gradient-accumulation work.

The proposed two-dimensional lookup keeps each rank's existing independent
batch. At equal per-rank token count it therefore trains on 32 times as many
unique tokens per group. Its extra routing cost is the price paid for retaining
that data-parallel throughput.

Column-only TP=32 also shards the per-head width `D`, not the final concatenated
`ple_embed_dim`. Very narrow `D / 32` shards may be invalid or inefficient even
when `ple_embed_dim` itself is divisible by 32.

Column-only PLE TP is preferable when one of the following is true:

- the entire surrounding model also adopts the same TP group, so dense compute
  is not replicated;
- inference or another workload intentionally uses one replicated request;
- reduced data-parallel throughput is acceptable and minimizing collective
  complexity is more important than training throughput.

For the current VeOmni training architecture, where global TP is unavailable
and each rank is expected to consume a different sample, the two-dimensional
layout is the preferred design.

## Parameter layout and ownership

Each checkpoint-native `ngram_embedding.shard_N.weight` has logical shape
`[V_N, D]`, where `V_N` includes the existing trailing row padding and `D` is
the embedding width per n-gram head.

The runtime parameter is a DTensor on the complete PLE mesh:

```python
device_mesh = parallel_state.extra_parallel_fsdp_device_mesh["ple"]
placements = [Shard(1), Shard(0)]  # mesh order: [ple_fsdp, ple]
local_shape = [V_N // P, D // F]
```

The following divisibility checks are mandatory:

```text
V_N % P == 0
D % F == 0
```

### Parallel-plan representation

The existing PLE plan remains row sharded:

```python
ple_plan = {
    "model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*.weight": Shard(0),
}
```

`ParallelPlan` and `SpecInfo` need an internal persistent-complementary-shard
marker, for example:

```python
SpecInfo(
    para_name="ple",
    placement=Shard(0),
    persistent_fsdp_shard_dim=1,
    ...,
)
```

The name describes an internal ExtraParallel policy, not a public TP API. A
separate `extra_parallel_persistent_modules` mapping should identify the PLE
embedding module. Do not overload `extra_parallel_fsdp_no_shard_module`: that
field currently identifies modules that are separately wrapped by FSDP2 on
the complementary mesh.

For a persistent parameter, `ParallelPlan.apply()` should create a DTensor
skeleton directly with `[Shard(1), Shard(0)]` and the correct global and local
shapes. It must not convert the DTensor back to a plain local parameter as the
current ExtraParallel path does before FSDP2 wrapping.

### FSDP2 ownership boundary

PLE embedding parameters must not belong to any FSDP2 parameter group.

`parallelize_model_fsdp2()` must:

1. Collect parameters marked as persistent PLE shards.
2. Skip the separate `fully_shard(ple_embedding, mesh=ple_fsdp_mesh)` call.
3. Pass the relevant persistent parameters through FSDP2's `ignored_params`
   argument for every ancestor `fully_shard()` call that can see them,
   including decoder-layer and root calls.
4. Continue applying ordinary FSDP2 to PLE projections, norms, convolution,
   decoder layers, and the root model.
5. Keep persistent PLE modules out of FSDP forward/backward prefetch lists.

The ignored set passed to a module should be intersected with that module's
parameters if required by the pinned PyTorch API. Both supported accelerator
stacks provide the FSDP2 `ignored_params` argument.

FSDP2 does not move or synchronize ignored parameters. The checkpoint loader
therefore owns their materialization, dtype conversion, and device placement.

## Distributed lookup

The PLE ranks continue to process different input samples. Consequently, a
standard tensor-parallel `all_gather(local_embedding, group=ple_group)` is
incorrect: each rank would contribute columns for different token IDs.

The lookup instead performs sparse two-dimensional request routing.

### Inputs

The existing hash and checkpoint-shard mapping are unchanged. Flatten the
lookup into `K` requests:

```text
shard_ids: [K]
row_ids:   [K]
```

For every request:

```python
row_owner = row_id // local_rows
local_row = row_id % local_rows
```

### Request routing

Each requested row has `F` column slices. Replicate the request metadata once
for every `col_owner` and route each copy to coordinate:

```text
(col_owner, row_owner)
```

The request payload contains the lookup information needed by the owner:

```text
(checkpoint_shard_id, local_row)
```

The source keeps the original-position/column permutation locally. The return
all-to-all preserves the per-destination order, so those two integers do not
need to cross the network.

Use variable-size `all_to_all_single` over a process group that flattens the
`ple_fsdp × ple` dimensions for the current replica. Request metadata is not
differentiable.

### Local lookup

Each destination owns `[V_N / P, D / F]` and performs:

```python
local_weight = embedding.weight.to_local()
local_output = torch.nn.functional.embedding(local_row, local_weight)
```

The local output has shape `[received_requests, D / F]`. Allocation and shape
logic must use the local weight width, not `nn.Embedding.embedding_dim`, which
still describes the logical global width.

PLE weights remain FP32 master parameters because FSDP2 ignores them. Under
mixed precision, cast `local_output` to the current hidden-state dtype before
the result all-to-all. This keeps communication and the downstream PLE
projections in the compute dtype without duplicating or converting the large
parameter shards.

### Result routing and reconstruction

Return the local vector slices to their source ranks with the existing
autograd-aware all-to-all helper. Reverse the request permutation and arrange
the result as:

```text
[K, F, D / F] -> [K, D]
```

The existing n-gram-head reshape then produces the unchanged PLE result:

```text
[batch, sequence, ple_embed_dim]
```

No vocabulary row or embedding table is gathered. Only requested vector
slices are returned.

### Communication cost

Ignoring count exchange and headers, each source rank communicates:

```text
request metadata: O(K * F) integers
returned values:  O(K * D) elements
parameter data:   0
```

The first implementation should use one direct variable-size all-to-all over
the flattened two-dimensional group. A hierarchical row/column implementation
may be benchmarked later, but it must preserve identical routing semantics.

## Backward and gradient averaging

The value-return all-to-all must be autograd-aware. Its backward exchanges each
`D / F` gradient slice back to the parameter owner. Embedding backward then
accumulates all requests targeting a local row.

Because the complete two-dimensional PLE group routes all data-rank requests
to the unique owner of each weight element, the owner's local gradient is the
sum of the participating ranks' gradients. FSDP2 no longer performs a
reduce-scatter or gradient divide for these parameters.

To match VeOmni's existing averaged-gradient semantics, install exactly one
gradient scaling point for persistent PLE parameters:

```python
grad /= parallel_state.extra_parallel_gradient_divide_factor("ple")
```

Under the supported initial topology this factor is `world_size`. The scaling
may be implemented as a parameter gradient hook or inside a dedicated lookup
autograd function. It must not be applied in both places.

Gradient accumulation continues to work because the scaling is applied to
each backward contribution in the same way as FSDP2's per-backward gradient
division.

## Hugging Face checkpoint loading

The released checkpoint format remains compatible. Each
`ngram_embedding.shard_N.weight` is a complete two-dimensional tensor for that
checkpoint-native row range; it is not a pre-sharded runtime-rank tensor.

The streaming loader must read the local rectangle directly:

```python
row_start = row_rank * local_rows
row_end = row_start + local_rows
col_start = col_rank * local_cols
col_end = col_start + local_cols

local = safetensor_slice[
    min(row_start, real_rows) : min(row_end, real_rows),
    col_start:col_end,
]
```

If the rank intersects the padded dim-0 tail, append zero rows until the local
shape is `[local_rows, local_cols]`.

Required loader changes:

- Generalize the current dim-0-only streaming metadata to use the parameter's
  declared placements and mesh coordinates.
- Support a two-dimensional `get_slice` without first materializing the full
  checkpoint tensor.
- Copy an already-local rectangle into the local storage of the persistent
  DTensor without redistributing or slicing it again.
- Preserve strict global-shape validation and converter-declared dim-0 padding.
- Generalize the non-streaming fallback's `ParallelPlan.shard_tensor()` to use
  the actual `Shard.dim`; keep streaming mandatory for the production 95 GiB
  checkpoint.

The Qwen4-Exp checkpoint converter should continue to own key recognition,
MTP filtering, and dim-0 padding semantics. Rank-specific row/column slicing
belongs to the loader, not the converter.

Column slicing a row-major safetensor is format-compatible but may have poorer
physical I/O locality than row slicing. The real-checkpoint deployment gate
must measure bytes read and startup time as well as host materialization size.

## DCP model and optimizer state

A persistent PLE parameter is already a DTensor on the complete mesh with
placements `[Shard(1), Shard(0)]`. It should be passed to DCP in that form.

The existing ExtraParallel DCP adapter drops the ExtraParallel dimension before
handing an FSDP-local tensor to DCP and restores it afterward. Extend
`SpecInfo` so the adapter can distinguish:

- ordinary ExtraParallel + FSDP2 parameters: retain the existing drop/restore
  behavior;
- persistent two-dimensional PLE parameters: skip drop/restore because the
  runtime parameter already carries the complete two-dimensional placement.

AdamW state tensors (`exp_avg` and `exp_avg_sq`) must retain the same mesh and
placements as their PLE parameter. Scalar optimizer state remains replicated.

Required DCP tests are:

- save and resume with the same `P × F` topology;
- reshard between different `P × F` factorizations at the same world size;
- load a model checkpoint written by the current FSDP-managed PLE layout into
  the persistent layout;
- either restore old optimizer state correctly or reject it with an explicit
  compatibility error before the optimizer step.

The old and new layouts describe the same logical global tensor and use the
same final `[Shard(1), Shard(0)]` DCP placement, so model-state compatibility is
expected and must be verified rather than assumed.

## Optimizer and gradient clipping

Persistent PLE parameters remain trainable DTensors and belong to the `ple`
ExtraParallel optimizer bucket. Optimizer classification should accept either
the DTensor mesh names or explicit `SpecInfo.para_name`; it must not depend on
the parameter having been wrapped by FSDP2.

AdamW updates only the local `[V_N / P, D / F]` parameter and local optimizer
state. No optimizer-time parameter gather is allowed.

The PLE shards are disjoint across both mesh axes. The global gradient norm
therefore reduces the local p-th-power sum exactly once over the flattened
`ple_fsdp × ple` process group. Persistent parameters must be split from the
ordinary ExtraParallel bucket; applying the usual sequential `ple_fsdp` then
`ple` reduction obscures their distinct ownership and creates an unnecessary
second collective boundary. Tests must confirm that persistent DTensors enter
this dedicated bucket and that the reported norm and clipped update match an
unsharded reference.

## Configuration

No new user-facing TP option is introduced. Existing configuration remains:

```yaml
train:
  accelerator:
    tp_size: 1
    ep_size: 4
    ulysses_size: 1
    cp_size: 1
    extra_parallel_names: [ple]
    extra_parallel_sizes: [8]
```

`ple_size` continues to mean the vocabulary-row degree. The complementary
`ple_fsdp_size` is derived as today but acts as the PLE column degree for the
embedding weights. Logs should report both roles explicitly:

```text
PLE 2D layout: rows=8 (ple), columns=8 (ple_fsdp), local=[V/8, E/8]
```

## Implementation map

| Area | Required change |
|------|-----------------|
| `veomni/distributed/parallel_plan.py` | Persistent complementary-shard metadata; construct/track 2D PLE DTensors. |
| `veomni/distributed/parallel_state.py` | Accessor for the flattened `ple_fsdp × ple` group and stable mesh-coordinate mapping. |
| `veomni/distributed/torch_parallelize.py` | Skip PLE `fully_shard`, propagate `ignored_params`, and exclude PLE from prefetch. |
| `veomni/models/module_utils.py` | Placement-aware 2D safetensor streaming and already-local DTensor dispatch. |
| `veomni/checkpoint/dcp_checkpointer.py` | Preserve persistent 2D DTensors instead of applying FSDP drop/restore. |
| `veomni/optim/optimizer.py` | Classify persistent PLE DTensors into the `ple` optimizer bucket. |
| `veomni/distributed/fsdp2/clip_grad_norm.py` | Verify two-axis norm reduction for persistent PLE shards. |
| `veomni/models/transformers/qwen4_exp/parallel_plan.py` | Declare the persistent PLE shard and the disjoint EP expert shard. |
| `qwen4_exp_*_patch_gen_config.py` | Implement GPU/NPU two-dimensional request/result routing. |
| `tests/models/test_qwen4_exp.py` | Lookup, gradient, loader, shape, and padding tests. |
| `tests/e2e/test_qwen4_exp_pipeline.py` | Optimizer and DCP resume coverage. |

Generated files under `veomni/models/transformers/qwen4_exp/generated/` must
only be changed by running patchgen.

## Validation plan

### Functional matrix

| World | `P` | `F` | `EP` | Purpose |
|-------|-----|-----|------|---------|
| 1 | 1 | 1 | 1 | Unsharded reference behavior. |
| 2 | 2 | 1 | 1 | Row-only distributed lookup. |
| 4 | 2 | 2 | 1 | Required two-dimensional PLE correctness case. |
| 2 | 2 | 1 | 2 | PLE+EP fused-MoE training and DCP resume smoke. |

Every distributed case must cover:

- different token IDs on every rank;
- unequal request counts;
- multiple checkpoint-native `shard_N` tables;
- repeated IDs and empty destination buckets;
- dim-0 padded rows;
- forward parity with a complete reference table;
- backward parity after reconstructing the global gradient;
- one AdamW update and reconstructed-weight parity.
- with EP enabled, finite non-zero expert gradients and a changed local expert
  shard after the optimizer step.

### Checkpoint tests

- Each streaming rank receives exactly `[V_N / P, D / F]`.
- No rank materializes `[V_N, D]`, `[V_N / P, D]`, or `[V_N, D / F]`.
- DCP model and optimizer save/resume preserve placements and values.
- Unsupported topology and loader combinations fail before reading large
  tensors.

### Performance acceptance

For a total PLE size `M`, peak PLE parameter memory per rank must be close to:

```text
M / (P * F)
```

plus selected-vector activations and routing workspace. For `M = 95 GiB`,
`P = 8`, and `F = 8`, the parameter component should be approximately
`1.5 GiB` per rank instead of the current approximately `11.9 GiB` forward
materialization.

A profiler or collective trace must show no PLE parameter all-gather. Dense
model FSDP all-gathers are expected and must be distinguished from PLE
communication.

The real-checkpoint deployment gate records:

- peak accelerator memory;
- peak host memory during loading;
- checkpoint bytes read per rank;
- PLE lookup communication time;
- end-to-end step time;
- forward/loss/gradient parity on a small deterministic batch.

## Delivery sequence

1. Add the persistent-shard plan metadata, mesh accessor, topology guards, and
   unit tests.
2. Add two-dimensional streaming load and local DTensor materialization tests.
3. Add the GPU and NPU PLE lookup implementation in patchgen sources and
   regenerate generated modeling files.
4. Add gradient scaling, optimizer classification, and gradient-norm parity
   tests.
5. Add DCP model/optimizer save-resume and topology-reshard tests.
6. Run the four-rank correctness suite and real-checkpoint memory/communication
   validation.
7. Update the Qwen4-Exp README, example configuration comments, architecture
   notes, and hard constraints.
8. Run the `/veomni-review` gate, `make quality`, focused distributed tests,
   and the Qwen4-Exp end-to-end regression before committing.
