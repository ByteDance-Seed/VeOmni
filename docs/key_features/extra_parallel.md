# Extra Parallelism

**Author**: Junncheng Wan

> TL;DR: VeOmni now supports extra parallelism from v0.1.0; Simply try it out by setting `model.accelerator.fsdp_config.fsdp_mode` to `fsdp2`, `model.accelerator.extra_parallel_sizes` to a list of integers, `model.accelerator.extra_parallel_placement_innermost` to a list of bools, and `model.accelerator.extra_parallel_names` to a list of strings.


## Motivation

VeOmni supports EP+FSDP2, but other modules, such as embedding layers, also need similar parallelism. To reuse the same communication operations, VeOmni generalizes EP+FSDP2 into ExtraParallel+FSDP2:

* Support a list of parallelism sizes of any length for different parallelism patterns in FSDP2 training.
* Support checkpoint save and (resharding) load for different parallelism patterns.
* Support prefetching to overlap communication and computation as described in [ep_fsdp2.md](./ep_fsdp2.md).

## Design Overview

The overall design of extra parallelism is similar to EP+FSDP2, except that it applies to different parallel modules. Before reading this document, read [ep_fsdp2.md](./ep_fsdp2.md). The key requirement is:

* The sharded modules need to be sorted in reverse order from submodules to parent modules to avoid sharding twice, as `fully_shard` is applied from bottom to top.

* In clipping gradient norm, individually judge the extra parallel mode of parameters and non-extra-parallel parameters.

### Sharding Dimension

In VeOmni, experts module is defined as tensors of [E, H, I] (Expert number, hidden dim, intermediate size) for down projection weights, and [E, I, H] for gate projection and up projection. Embedding is defined as tensors of [V, H] (Vocab size, hidden dim).

> please see [modeling_qwen3_moe.py](../../veomni/models/transformers/qwen3_moe/modeling_qwen3_moe.py) for detailed implementation of experts and embedding layer.

Extra parallelism is applied on dim-0 (expert number, vocab size), while FSDP2 is applied on dim-1 instead of default dim-0 for more flexible parallelism setup. Otherwise, if we also choose dim-0 for FSDP2, Expert Parallel or Embed Parallel x FSDP2 size needs to be exact expert number or vocab size.

The lookup of a vocab-sharded embedding is described in [sharded_embedding.md](./sharded_embedding.md).

## Usage

> File: tests/utils/test_extra_parallel.py

When using train script (e.g. [tasks/train_vlm.py](../../tasks/train_vlm.py)), add the arguments:
```shell
--model.accelerator.extra_parallel_sizes size1 size2
--model.accelerator.extra_parallel_placement_innermost bool1 bool2
--model.accelerator.extra_parallel_names name1 name2
```

In the parallel plan config (e.g. [qwen3_moe/parallel_plan.py](../../veomni/models/transformers/qwen3_moe/parallel_plan.py)), add
```python
ep_plan = {
    "model.layers.*.mlp.experts.gate_proj": Shard(0),
    "model.layers.*.mlp.experts.up_proj": Shard(0),
    "model.layers.*.mlp.experts.down_proj": Shard(0),
}
extra_parallel_1_plan = {
    ...
}
extra_parallel_2_plan = {
    ...
}
parallel_plan = ParallelPlan(
    extra_parallel_plan={
        "ep": ep_plan,
        "extra_parallel_1": extra_parallel_1_plan,
        "extra_parallel_2": extra_parallel_2_plan,
    }
)
```

### Plan owners

Each plan key's parent module is that key's owner. `ParallelPlan.__init__` registers
every distinct owner pattern in `extra_parallel_fsdp_no_shard_module[<name>]`, and the
parallelizer wraps each matched owner as its own FSDP2 unit on that dimension's
`<name>_fsdp` mesh. For the plan above, `model.layers.*.mlp.experts` is the `ep` owner.
Do not set `extra_parallel_fsdp_no_shard_module` by hand.

* Owners must not nest. A plan with both `decoder.weight` and `decoder.embed.weight`
  raises `ValueError`, because each owner would become its own unit.
* Every owner must sit under a module whose class is listed in the model's
  `_no_split_modules`. The parallelizer pairs owners only under those wrap targets.
  An owner outside them is still sliced by the plan, but FSDP2 shards it again over
  the regular mesh. Each rank then gathers a mix of other ranks' slices, and nothing
  raises. A MoE MTP head is the usual case: see step 6 in [mtp.md](./mtp.md).

### Inference without the FSDP wrap

ExtraParallel normally composes with FSDP2, and the two shard **different
dimensions**. For an expert stack `[E, H, I]` the plan takes dim0
(`Shard(0)` over experts) and the wrap takes dim1 by default — see the
`shard_placement_fn` that hands FSDP `Shard(1)`, or dim0 again when
`expert_fsdp_shard_dim: 0` asks for whole-expert shards. `<para>_fsdp` names
that second axis: with `ep_size=64` on 256 cards it is 4.

At inference the second axis often has nothing to do, and for a checkpoint kept
quantized it cannot run at all — an FSDP parameter group requires a single
original dtype, while a decoder layer holding quantized storage as released has
several (packed int8 weights, fp8 values, fp8 or fp32 scales, fp32 numerics
buffers). That is the *inference* reason. A training run dequantizes at load —
allocating bf16 (or fp32, with the wrap) parameters and consuming the scales in
the checkpoint converter — so its layer holds one dtype and FSDP2 accepts it.

Because FSDP2's `MixedPrecisionPolicy` is part of the skipped wrap,
`extra_parallel_only: true` only applies when `fsdp_mode: fsdp2` (ignored under
`ddp` or `eager`) and requires `mixed_precision.enable: false` (otherwise
`build_parallelize_model` would upcast weights to FP32 with no policy to cast
them back to the compute dtype, breaking BF16-only kernels such as Triton MoE
GEMMs). Inference defaults `mixed_precision.enable` to `false`; a training run
must set it explicitly:

```yaml
accelerator:
  ep_size: 32                 # == the shard axis, so <para>_fsdp is 1
  fsdp_config:
    fsdp_mode: fsdp2
    extra_parallel_only: true
    mixed_precision:
      enable: false           # required with extra_parallel_only (also avoids the fp32 master copy)
```

Large row-sharded embedding tables (such as `emb`-parallel `ShardedEmbedding`
modules) can use this setup to avoid both FSDP all-gather buffers and an extra
FP32 master copy.

This applies the plan and skips the wrap. The two shard for different reasons:
FSDP shards for *storage* and all-gathers the parameter back before the
operator runs, whereas ExtraParallel shards for *compute* and never reassembles
— tokens are routed to the rank holding their expert or their table rows. Only
the second is load-bearing when there are no gradients and no optimizer state.

It is required rather than merely helpful when `<para>` already spans the whole
shard axis: FSDP then shards nothing but still allocates an all-gather buffer
the size of the local shard. Because the plan leaves the `<para>_fsdp` axis to
FSDP, the mode refuses a topology where that axis is wider than one rank —
without the wrap it would not be sharded at all, and every expert would be
replicated `<para>_fsdp` times in silence. In practice that means `ep_size` must
equal the shard width (or an `emb` width must equal the card count) for the
dimension to qualify.

Note that inference defaults `mixed_precision.enable` to `false`. Mixed
precision keeps fp32 master weights for an optimizer; left on it builds the
model at float32 and calls `model.float()`, which upcasts quantized fp8 weights
and scales along with everything else.

## Acknowledgements

Big thanks to ByteDance Seed team: Bin Jia, Zheng Zhang, Yifan Pi, Tianle Zhong, Zhelun Shi, Zhi Zhang
