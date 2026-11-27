# FSDPTurbo

[FSDPTurbo](https://gitcode.com/Ascend/FSDPTurbo) is an optional FSDP2 communication
and prefetch acceleration backend integrated through a reversible VeOmni adapter.
It inherits VeOmni's native FSDP unit partitioning (`_no_split_modules`, `ExtraParallel`,
and mixed-precision exclusions) while preserving native PyTorch `FSDPModule` and `DTensor`
semantics across weight loading, DCP checkpointing, gradient clipping, LoRA export, and
optimizers. This integration supports the Text and VLM trainers.

## Install dependencies

Install `fsdp-turbo` into your VeOmni environment alongside your target accelerator extra:

```bash
pip install -e /path/to/FSDPTurbo
```

Installing `fsdp-turbo` does not enable it automatically; select `fsdp_mode: turbo_fsdp2`
in your training YAML.

## Configure the backend

Add the following settings to your training YAML:

```yaml
model:
  accelerator:
    init_device: meta
    fsdp_config:
      fsdp_mode: turbo_fsdp2
      reshard_after_forward: true
      reshard_after_backward: true
      forward_prefetch: true
      fsdp_turbo:
        forward_prefetch_depth: 1
        backward_prefetch_depth: 1
        max_inflight_all_gathers: 2
        max_inflight_reductions: 2
        prefetch_wait_a2a: false
        backward_prefetch_on_moe_input: false
```

- `forward_prefetch_depth` (`>= 0`): number of following blocks to prefetch during forward (`0` disables forward prefetch; overridden to `0` when `fsdp_config.forward_prefetch` is `false`).
- `backward_prefetch_depth` (`>= 0`): number of preceding blocks to prefetch during backward (`0` disables all backward prefetch, including PyTorch FSDP2's implicit 1-unit default).
- `max_inflight_all_gathers` (`>= 1`): `InflightQueue` cap for concurrent All-Gather copy-outs.
- `max_inflight_reductions` (`>= 1`): `InflightQueue` cap for concurrent Reduce-Scatter / gradient reductions.
- `prefetch_wait_a2a`: order forward/backward prefetch All-Gather streams behind the latest MoE expert All-to-All completion event (requires `ep_size > 1`).
- `backward_prefetch_on_moe_input`: issue MoE-layer backward parameter prefetches from the MoE input tensor gradient hook right after reverse dispatch All-to-All (requires `ep_size > 1` and `backward_prefetch_depth >= 1`).

`turbo_fsdp2` requires `model.accelerator.init_device: meta` and cannot be combined with
`torch_compile.enable: true`, `fsdp_config.extra_parallel_only: true`,
`fsdp_config.low_precision_reduce_scatter_comm: true`, or
`fsdp_config.expert_weight_fp8_all_gather: true`.

Normal trainer teardown drains pending DCP saves and removes the FSDPTurbo patch.

## Adapter patch targets

`fsdp_turbo.integration.veomni` rebinds at most three VeOmni attributes:

| Binding / helper | Purpose |
| --- | --- |
| `veomni.distributed.torch_parallelize.fully_shard` | Delegate FSDP wrapping to `fine_grained_fully_shard.fully_shard` with inflight queue limits while preserving root `reshard_after_forward=False`. |
| `veomni.distributed.torch_parallelize.parallelize_model_fsdp2` | Configure multi-depth forward/backward prefetch, zero-depth backward override, and MoE prefetch hooks. |
| `veomni.distributed.moe.moe_layer.all_to_all` | Record expert All-to-All completion events when `prefetch_wait_a2a: true`. |
| `veomni.distributed.torch_parallelize.sort_fqn_by_submodule_first` | Order target blocks within each container prefix (called, not replaced). |
