# Metric Meter (`MetricMeterMixin`)

`MetricMeterMixin` lets a SeedOmni module report its own tokens and theoretical
FLOPs, so `OmniTrainer` can log MFU, achieved FLOPs and per-module token
throughput.

- Mixin: [`veomni/models/seed_omni/mixins/metric_meter_mixin.py`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/seed_omni/mixins/metric_meter_mixin.py)
- Roll-up: [`OmniEnvironMeter`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/utils/omni_helper.py)
- Trainer hook: [`OmniStepMetricsCallback`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py)

## Why each module meters itself

The single-model trainers count tokens from `input_ids` and pick one FLOPs
formula from the model's `model_type` (`EnvironMeter` +
`VeomniFlopsCounter`). An `OmniModel` is a composition of independent modules
(text encoder, vision encoder, AR backbone, VQ codec, ...). It has no single
config to dispatch a FLOPs formula on, and its batch only carries
`conversation_list`. A whole-model estimator is also wrong at module
granularity: an AR backbone owns no `wte` / `lm_head`, those FLOPs belong to
the text-encoder module.

So the work is split:

| Who | Produces | Time-dependent? |
|-----|----------|-----------------|
| Each metered module | Its own per-sample token lengths and its own theoretical FLOPs | No |
| `OmniEnvironMeter` | Cross-module, cross-rank sums divided by the whole-step wall-clock: achieved FLOPs, MFU, tokens/s | Yes |

Modules never time themselves. They share one backward and their compute
interleaves inside the step, so a per-module wall-clock has no meaning; the
trainer times the whole step once.

## Opting in

Metering is opt-in through multiple inheritance. `BaseMixin` does **not**
inherit `MetricMeterMixin`, and modules that do not mix it in contribute nothing.
Add it next to the module's other VeOmni mixins and implement `estimate_flops`.
Call `metric_meter_set_seqlens` from the module's `pre_forward` hook:

```python
from typing import Any

from veomni.models.seed_omni.mixins import (
    BaseMixin,
    InferenceModuleMixin,
    MetricMeterMixin,
    TrainingModuleMixin,
    pre_forward,
)


class MetricMeter(MetricMeterMixin):
    def estimate_flops(self, seqlens: list[int]) -> float:
        # Forward + backward for the dense linears (6 * params * tokens), in TFLOPs.
        # `num_linear_params` stands in for whatever the module's own formula needs.
        return 6 * self.num_linear_params * sum(seqlens) / 1e12


class TrainingMixin(TrainingModuleMixin):
    @pre_forward("forward")
    def forward_pre(self, input_ids: list[Any], **kwargs: Any) -> dict[str, Any]:
        # Full per-sample lengths, BEFORE any sequence-parallel slice.
        self.metric_meter_set_seqlens("forward", [len(ids) for ids in input_ids])
        ...  # pack / pad / SP-slice, then return the endpoint kwargs


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin, MetricMeter):
    pass
```

## The functions

### What a module implements or calls

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| `estimate_flops(seqlens) -> float` | **Must implement.** Return the module's total theoretical TFLOPs (forward + backward) for the given token lengths. | `metric_meter_collect`, once per global step | Summed with the other modules' FLOPs into one `flops_achieved(T)` / `mfu`. Not implementing it raises `NotImplementedError` at the first step end. |
| `metric_meter_set_seqlens(method, seqlens)` | **Must call** inside the `pre_forward` hook of every call-site whose tokens should count, before any SP gather/slice. | The module's own `pre_forward` | Stashes the full per-sample lengths under `method` until the executor drains them. A call-site that never stashes (for example a VQ codec's `decode`) is not counted. |
| `metric_meter_token_lengths(method, data) -> list[int]` | Optional override. The default pops the stash for `method` and ignores `data`. | `metric_meter_add` | Defines what gets accumulated. Overriding it to read `data` is possible, but `data` is the post-`pre_forward` kwargs, which under SP is only this rank's shard. |

### What the framework provides (do not override)

| Function | Called by | Effect |
|----------|-----------|--------|
| `MetricMeterMixin.metric_meter_add(method, data)` | `execute_train_node` in [`accelerated/utils/executor.py`](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/seed_omni/accelerated/utils/executor.py), once per training node, right after `pre_forward` and before the endpoint runs, inside the module's `ParallelState` scope | Appends `metric_meter_token_lengths(method, data)` to the module's per-step buffer. Over gradient accumulation it runs once per micro-batch, so the buffer holds the whole global step. |
| `MetricMeterMixin.metric_meter_collect() -> (flops, seqlens)` | `OmniModelRuntime.metric_meter_collect` | Returns `(estimate_flops(buffer), buffer)` and empties the buffer for the next step. |
| `OmniModelRuntime.metric_meter_collect() -> {name: (flops, seqlens)}` | `OmniStepMetricsCallback.on_step_end` | Drains every module with `isinstance(omni_module, MetricMeterMixin)`. Keys are module names, so they are identical on every rank even when a rank's batch did not use a module (it reports `(0.0, [])`). |
| `OmniEnvironMeter.add(micro_batch)` | `OmniStepMetricsCallback.on_step_begin`, once per micro-batch | Counts samples (`len(conversation_list)`) and, for multi-source, gathers `ds_idx`. Never looks at tokens. |
| `OmniEnvironMeter.step(delta_time, global_step, module_metrics)` | `OmniStepMetricsCallback.on_step_end` | One DP all-reduce of FLOPs, sample count and per-module token sums, then the metrics listed below. |

## Lifecycle of one training step

```text
OmniTrainer.train_step
├─ on_step_begin
│   └─ OmniStepMetricsCallback.on_step_begin
│       ├─ OmniEnvironMeter.add(micro_batch)            # per micro-batch: sample count, ds_idx
│       └─ start_time = time.time()
├─ for each micro_batch: forward_backward_step
│   └─ OmniModel.forward → TrainNodeRunner → execute_train_node(node)
│       ├─ raw.pre_forward(method, **batch)
│       │   └─ self.metric_meter_set_seqlens(method, full_lengths)   # module code, pre-SP-slice
│       ├─ raw.metric_meter_add(method, inputs)         # pops the stash into the step buffer
│       └─ endpoint → post_forward
├─ clip_grad_norm / optimizer.step / lr_scheduler.step
└─ on_step_end
    └─ OmniStepMetricsCallback.on_step_end
        ├─ delta_time = time.time() - start_time
        ├─ module_metrics = OmniModelRuntime.metric_meter_collect()
        │   └─ per metered module: (estimate_flops(buffer), buffer); buffer reset
        └─ OmniEnvironMeter.step(delta_time, global_step, module_metrics)
            → trainer.step_env_metrics  (logged by WandbTraceCallback)
```

`delta_time` covers every micro-batch's forward and backward plus the optimizer
step. Metering only happens through `TrainNodeRunner`: calling
`OmniModel.forward` without a `node_runner` (the eager, framework-free path)
runs `pre_forward` but never drains the stash, so nothing is counted there.

## Resulting metrics

`OmniEnvironMeter.step` returns these keys; `OmniStepMetricsCallback` merges the
`training/*` losses into the same dict and `WandbTraceCallback` logs it.

| Key | Meaning |
|-----|---------|
| `flops_achieved(T)` | Sum over modules and DP ranks of `estimate_flops`, divided by `delta_time`. |
| `flops_promised(T)` | Device peak TFLOPs × world size. |
| `mfu` | `flops_achieved / flops_promised`. |
| `consumed_chunk_num` | Cumulative real training samples (conversations) across DP ranks. |
| `trace/<module>/tokens_per_second(M)` | This module's global tokens this step / `delta_time`. |
| `trace/<module>/consume_tokens(M)`, `(B)` | This module's cumulative global tokens. |
| `trace/<module>/avg_seq_len` | This module's global tokens this step / `global_batch_size`. |
| `max_memory_allocated(GB)`, `cpu_used_memory(GB)`, ... | Device and host memory, reduced to the worst rank (shared with `EnvironMeter`). |
| `multi_source/...` | Only with `data.enable_multisource`; see below. |

`<module>` is the module's name in the `OmniModel` (the key of
`OmniModelRuntime.module_runtimes`). While no module is metered, the three FLOPs
keys are omitted rather than logged as zero; token and memory metrics still
appear.

FLOPs are summed across modules because each module's compute is distinct
(backbone layers vs. ViT vs. `lm_head`). Tokens are **not** summed: the
backbone's sequence already contains the text and image tokens the encoder
modules also count, so a merged total would double-count. That is why tokens are
reported per module under `trace/<module>/` and there is no top-level
`tokens_per_second(M)`.

`consume_tokens` is part of `OmniEnvironMeter.state_dict()`, which
`GlobalStateCallback` saves and restores, so the cumulative counters survive a
resume.

## Rules for a correct implementation

- **Stash before the SP slice.** Under sequence parallelism `pre_forward`
  slices to this rank's `1/sp_size` shard. Stash the full per-sample lengths
  before that branch. `OmniEnvironMeter` reduces over the `dp_group`, which
  excludes the SP peers that hold the same sample, so each sample is counted
  once and the totals match a non-SP run. Measuring after the slice
  under-counts by about `sp_size`.
- **One token domain per module.** A module's lengths are whatever its compute
  scales with (text tokens, image patches, VQ codes). `estimate_flops` receives
  exactly the list the module stashed.
- **`estimate_flops` covers forward + backward and returns TFLOPs.** It must
  return `0.0` for `[]`, because a rank whose batch skipped the module still
  collects it. Count only parameters the module owns; do not reuse
  `VeomniFlopsCounter`'s whole-model formula.
- **Stash once per call-site invocation.** The stash is keyed by `method` and
  popped right after `pre_forward`, so re-reading it cannot double-count, and a
  module invoked by several nodes is counted once per invocation.
- **Count the compute the module actually does.** Frozen modules are drained
  like any other metered module. If no gradient flows through a frozen module,
  it only runs a forward, and its `estimate_flops` should not include a
  backward term.

### Multi-source accounting

With `data.enable_multisource`, the multi-source tracker needs one sequence
length per sample to attribute tokens to each dataset. `OmniEnvironMeter`
picks the source module collectively: among the modules whose step buffer has
exactly one entry per sample (`len(seqlens) == len(ds_idx)`) on **every** DP
rank, it takes the one with the most global tokens, normally the backbone, whose
sequence is the union of all modalities. Ties resolve by name. If no module
qualifies, multi-source token counts are logged as zero with a one-time warning.
To make a backbone eligible, stash exactly one length per conversation.

## Tests

[`tests/seed_omni/mixins/test_metric_meter_mixin.py`](https://github.com/ByteDance-Seed/VeOmni/blob/main/tests/seed_omni/mixins/test_metric_meter_mixin.py)
covers the mixin's stash/accumulate/reset contract and the `OmniEnvironMeter`
roll-up (including a two-rank gloo run). The runtime drain is covered in
`tests/seed_omni/runtime/test_omni_model_runtime.py`, the callback wiring in
`tests/seed_omni/trainer/test_step_metrics_callback.py`.
