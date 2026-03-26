# Extra Parallelism

**Author**: Junncheng Wan

> TL;DR: VeOmni now supports extra parallelism from v0.1.0; Simply try it out by setting `accelerator.fsdp_config.fsdp_mode` to `fsdp1` or `fsdp2`, `accelerator.extra_parallel_sizes` to a list of integers, `accelerator.extra_parallel_placement_innermost` to a list of bools, and `accelerator.extra_parallel_names` to a list of strings.


## Motivation

As EP+FSDP1/FSDP2 is well supported in VeOmni, similar parallelism is also needed for other modules, like embedding layer. To support this kind of parallelism with similar communication ops, we extend EP+FSDP1/FSDP2 to extra parallelism+FSDP1/FSDP2:

* Support any length of list of parallelism sizes for different parallism patterns in FSDP1/FSDP2 training.
* Support checkpoint save and (resharding) load for different parallelism patterns.
* Support prefetching to overlap communication and computation as [ep_fsdp2.md](./key_features/ep_fsdp2.md).

## Design Overview

The overall design of extra parallelism is similar to EP+FSDP1/FSDP2, except that it is applied on different parallel modules. Before reading this document, please read [ep_fsdp2.md](./key_features/ep_fsdp2.md). There are several main changes as follow:

* For FSDP1, determine which modules in basic_modules, typically no split modules, are siblings nodes of fsdp_no_shard_states_fqn; if so, exclude these modules from the FSDP sharding of the entire model's root node at the initial stage to prevent them from being sharded twice later.

* For FSDP2, the sharded modules need to be sorted in reverse order from submodules to parent modules to avoid sharding twice, as full_shard is applied from bottom to top.

* In clipping gradient norm, individually judge the extra parallel mode of parameters and non-extra-parallel parameters.

### Sharding Dimension

In VeOmni, experts module is defined as tensors of [E, H, I] (Expert number, hidden dim, intermediate size) for down projection weights, and [E, I, H] for gate projection and up projection. Embedding is defined as tensors of [V, H] (Vocab size, hidden dim).

> please see [modeling_qwen3_moe_foundation.py](../../veomni/models/seed_omni/foundation/qwen3_moe_foundation/modeling_qwen3_moe_foundation.py) for detailed implementation of experts and embedding layer.

Extra parallelism is applied on dim-0 (expert number, vocab size), while FSDP2 is applied on dim-1 instead of default dim-0 for more flexible parallelism setup. Otherwise, if we also choose dim-0 for FSDP2, Expert Parallel or Embed Parallel x FSDP2 size needs to be exact expert number or vocab size.

## Usage

> File: tests/utils/test_extra_parallel_clip_grad_norm.py

When using train script (e.g. [tasks/train_vlm.py](../../tasks/train_vlm.py)), add the arguments:
```shell
--train.accelerator.extra_parallel_sizes size1 size2
--train.accelerator.extra_parallel_placement_innermost bool1 bool2
--train.accelerator.extra_parallel_names name1 name2
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



## Acknowledgements

Big thanks to ByteDance Seed team: Bin Jia, Zheng Zhang, Yifan Pi, Tianle Zhong, Zhelun Shi, Zhi Zhang
