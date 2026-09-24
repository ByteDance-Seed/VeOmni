# MoE Router Monitor and Expert Load Heatmaps

## Background

Mixture-of-Experts models select a small set of experts for each token. Uneven
selection can leave some experts idle while others carry most work. The router
monitor records actual expert selections and exposes per-layer imbalance metrics
and a heatmap. It is an observation tool: it does not change routing, add a
balancing loss or measure expert execution time.

Use an already working [MoE training recipe](../examples/qwen3_moe.md) and
[hardware installation](../get_started/installation/install.md).

## Features

### Expert selection counting

The router decides which experts receive a token. The monitor reads decisions
the model has already made and counts selections; it does not route tokens again.
If one token selects experts 0 and 2, each expert receives one count. Selecting k
experts contributes k selections.

```text
Model selects experts → accumulate counts per layer → reach reporting step
                     → sum participating ranks' counts → produce ratios/heatmap
                     → reset counts and start the next window
```

A reporting window includes multiple forwards and micro-batches. With an interval
of 100 in a normal run starting from scratch, the first report covers observations
from steps 1–100, and the next window starts at step 101. The heatmap caption shows
the corresponding range.

### Load normalization and imbalance metrics

Each row is a registered MoE layer and each column is an expert. A cell shows the
expert's share of all selections in that layer, not its execution time. Each row
with observations sums to 1.

Suppose a layer has two experts, selected 75 and 25 times during one window:

| Quantity | Expert 0 | Expert 1 |
| --- | --- | --- |
| Actual selection share | 75% | 25% |
| Share under uniform routing | 50% | 50% |
| Deviation relative to uniform | `(75% - 50%) / 50% = +0.5` | `(25% - 50%) / 50% = -0.5` |

Here, 0.5 means 50% above the uniform value, not 50 percentage points above it.
Three metrics summarize a layer's deviations:

| Metric | How to read it | Example result |
| --- | --- | --- |
| `max_vio` | Largest deviation: how far the busiest expert exceeds uniform routing. | `0.5` |
| `min_vio` | Smallest deviation: how far the least selected expert falls below uniform routing. | `-0.5` |
| `avg_vio` | Mean absolute deviation: the overall departure from uniform routing. | `0.5` |

For E experts, let `C[l,e]` be the count and
`P[l,e] = C[l,e] / max(sum_e C[l,e], 1)` the share. The relative deviation is
`D[l,e] = E * P[l,e] - 1`. The three metrics are the maximum, minimum and mean
absolute value of each row of D.

Uniform routing with observations gives zero for all three. A registered layer
without observations has a zero row, producing `max_vio=-1`, `min_vio=-1` and
`avg_vio=1`. This means **no observations**, not balanced routing or measured load.

### Aggregation across parallel ranks

A rank is a training process. Data-parallel (DP) processes handle different data,
and sequence-parallel (SP) processes handle different sequence slices, so their
observations must be combined. Every participating rank executes the reduction,
which sums their counts; only publishing the result can be rank-0-only.

The current callback supplies `parallel_state.fsdp_group` as the DP+SP aggregation
group. It does not perform an additional expert-parallel (EP) group reduction,
which would treat replicated gate observations as more distinct tokens. Although
the generic argument help mentions EP/DP, this call chain determines the actual
communication scope.

### Monitor integration and counting scope

`MoERouterMonitorCallback` creates the monitor when the interval is positive and
the model configuration exposes `num_experts`. At training start it attaches
supported routers and registers their order; if none are recognized, it warns and
disables monitoring. Most supported routers expose selected indices through a
hook; DeepSeek-V3 records them explicitly from its patched MoE block. Detached
`torch.bincount` operations accumulate counts on-device without contributing to
optimization.

The unit being counted is an observed expert selection, not a deduplicated
supervised token. Gradient-checkpoint recomputation may also pass through the
monitor, so counts are not a census of unique training samples or tokens.

## Configuration and usage

Merge this fragment into an existing supported MoE recipe:

```yaml
train:
  moe_load_balance_monitor_interval: 100
  wandb:
    enable: true
    project: veomni-router-monitor
    name: router-load-check
```

This is a **configuration fragment**, not a complete model/data configuration.
Use the recipe's existing launch command after merging. W&B must be configured
for your intended online or offline mode before training.

| Field | Default | Behavior |
| --- | --- | --- |
| `train.moe_load_balance_monitor_interval` | `0` | Disabled at zero; the callback also disables for negative values. Use a positive integer N to report every N optimizer steps. |
| `train.wandb.enable` | `false` | Required for the standard callback to publish scalars and images. Monitoring itself can be enabled without W&B. |

Run through at least one reporting boundary. Confirm startup reports attached
router modules and inspect these W&B keys:

* `moe/expert_load_heatmap`: expert columns, registered-router layer rows;
* `moe/{max,min,avg}_vio/layer_<i>`: per-layer metrics;
* `moe/{max,min,avg}_vio/{max,avg}`: across-layer aggregates.

Only global rank 0 publishes, but **all ranks must execute metric computation**.
In particular, `moe/min_vio/max` is the maximum of the per-layer minimum values;
it is not the most negative deficit across all layers.
