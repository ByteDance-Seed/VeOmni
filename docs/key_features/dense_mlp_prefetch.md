# Dense MLP prefetch

This opt-in FSDP2 optimization shards each selected decoder MLP before its parent block. The parent group owns the remaining parameters. Forward prefetch can overlap the MLP parameter gather with attention computation, and optionally gather the next parent group's parameters during the current MLP computation.

## Settings

Set these environment variables before launching training, consistently on every rank:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VEOMNI_DENSE_MLP_LAYERS` | Empty (disabled) | Path from the model passed to FSDP2 to a nonempty decoder `nn.ModuleList`, for example `model.language_model.layers`. Setting this enables the MLP split. |
| `VEOMNI_DENSE_MLP_PREFETCH` | `1` | Prefetch each block's MLP from the parent block's forward boundary. `0` disables explicit prefetch but retains the split. |
| `VEOMNI_DENSE_MLP_PREFETCH_NEXT` | `0` | With MLP prefetch enabled, `1` also prefetches the next parent's non-MLP parameters from the current MLP boundary. |

Both prefetch variables accept only `0` or `1`. They take effect only when a layer path is set. The existing `model.accelerator.fsdp_config.forward_prefetch` setting is the master switch: when false, neither explicit prefetch schedule is installed, but the split remains.

Each selected block must already be an FSDP wrapping target and contain a parameterized dense `.mlp` module. The MLP must not contain existing FSDP modules, other wrapping targets, or an `experts` attribute, and must not share parameters outside its group. The decoder must call the MLP normally; next-block prefetch assumes execution follows the `ModuleList` order.

## Enable or disable

Enable both schedules with `forward_prefetch: true`:

```bash
export VEOMNI_DENSE_MLP_LAYERS=model.language_model.layers
export VEOMNI_DENSE_MLP_PREFETCH=1
export VEOMNI_DENSE_MLP_PREFETCH_NEXT=1
```

For split-only behavior, keep the layer path and set `VEOMNI_DENSE_MLP_PREFETCH=0`. This also disables next-block prefetch, even if `VEOMNI_DENSE_MLP_PREFETCH_NEXT=1`.

To fully disable the feature and restore the original grouping, unset the layer path before launching a new training run:

```bash
unset VEOMNI_DENSE_MLP_LAYERS
unset VEOMNI_DENSE_MLP_PREFETCH
unset VEOMNI_DENSE_MLP_PREFETCH_NEXT
```

## Supported combinations

The split requires FSDP2 with compilation, tensor parallelism, ExtraParallel, and mixed-precision module exclusions disabled.

Dense MLP prefetch is rejected when selective activation checkpointing (SAC) is active. Use ordinary checkpointing with `selective_recompute_layers: 0`, or disable explicit MLP prefetch. Disabling prefetch retains the nested split; unset the layer path to remove it entirely.

Ordinary non-reentrant full checkpointing is not rejected by this guard. Nested FSDP recomputation still depends on runtime lifecycle support; this optimization does not backport fixes for older runtimes. The reported full-checkpointing runs used PyTorch `2.10.0+cpu`, torch_npu `2.10.0.post2`, and CANN `9.1.0`. Checkpoint save/resume was not validated by those runs.
