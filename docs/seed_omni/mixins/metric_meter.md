# Metric Meter (`MetricMeterMixin`)

`MetricMeterMixin` lets a SeedOmni module report its own tokens and theoretical
FLOPs, so `OmniTrainer` can log MFU, achieved FLOPs and per-module token
throughput.

- Mixin: [`MetricMeterMixin`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L76)
- Roll-up: [`OmniEnvironMeter`](../../../veomni/utils/omni_helper.py#L50)
- Trainer hook: [`OmniStepMetricsCallback`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L29)

## Why each module meters itself

The single-model trainers count tokens from `input_ids` and pick one FLOPs
formula from the model's `model_type`
([`EnvironMeter`](../../../veomni/utils/helper.py#L204) +
[`VeomniFlopsCounter`](../../../veomni/utils/count_flops.py#L73)). An
`OmniModel` is a composition of independent modules (text encoder, vision
encoder, AR backbone, VQ codec, ...). It has no single config to dispatch a
FLOPs formula on, and its batch only carries `conversation_list`. A whole-model
estimator is also wrong at module granularity: an AR backbone owns no `wte` /
`lm_head`, those FLOPs belong to the text-encoder module.

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
Add it next to the module's other VeOmni mixins (the
[`VeOmniMixin`](../../../veomni/models/seed_omni/modules/fake_model/fake_module_a/accelerated/accelerated.py#L47)
in the module's `accelerated/accelerated.py`) and implement `estimate_flops`.
Call `metric_meter_set_seqlens` from the module's
[`@pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L26)
hook:

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
        packed_input_ids = ...  # pack / pad / SP-slice
        # The returned dict becomes the endpoint's kwargs.
        return {"input_ids": packed_input_ids, **kwargs}


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin, MetricMeter):
    pass
```

## The functions

### What a module implements or calls

| Function | Module's job | Called by | Effect |
|----------|--------------|-----------|--------|
| [`estimate_flops(seqlens) -> float`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L105) | **Must implement.** Return the module's total theoretical TFLOPs (forward + backward) for the given token lengths. | [`metric_meter_collect`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L139), once per global step | Summed with the other modules' FLOPs into one `flops_achieved(T)` / `mfu`. Not implementing it raises `NotImplementedError` at the first step end. |
| [`metric_meter_set_seqlens(method, seqlens)`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L79) | **Must call** inside the `pre_forward` hook of every call-site whose tokens should count, before any SP gather/slice. A hook registered for several call-sites (`@pre_forward("encode", "offline_encode")`) cannot tell which one invoked it, so it stashes under each. | The module's own `pre_forward`, dispatched by [`TrainingModuleMixin.pre_forward`](../../../veomni/models/seed_omni/mixins/training_module_mixin.py#L59) | Stashes the full per-sample lengths under `method` until the executor drains them. A call-site that never stashes (for example a VQ codec's `decode`) is not counted. |

### What the framework provides (do not override)

| Function | Called by | Effect |
|----------|-----------|--------|
| [`MetricMeterMixin.metric_meter_add(method)`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L120) | [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L66), once per training node, right after `pre_forward` and before the endpoint runs, inside the module's `ParallelState` scope | [Appends](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L136) the lengths stashed for the running `method` to the module's per-step buffer, then [clears](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L137) the whole stash, so copies a shared hook left under other call-sites cannot leak into a later node. Over gradient accumulation it runs once per micro-batch, so the buffer holds the whole global step. |
| [`MetricMeterMixin.metric_meter_collect() -> (flops, seqlens)`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L139) | [`OmniModelRuntime.metric_meter_collect`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L546) | Returns `(estimate_flops(buffer), buffer)` and empties the buffer for the next step. |
| [`OmniModelRuntime.metric_meter_collect() -> {name: (flops, seqlens)}`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L546) | [`OmniStepMetricsCallback.on_step_end`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L76) | Drains every module with `isinstance(omni_module, MetricMeterMixin)`. Keys are module names, so they are identical on every rank even when a rank's batch did not use a module (it reports `(0.0, [])`). |
| [`OmniEnvironMeter.add(micro_batch)`](../../../veomni/utils/omni_helper.py#L123) | [`OmniStepMetricsCallback.on_step_begin`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L66), once per micro-batch | Counts samples (`len(conversation_list)`) and, for multi-source, gathers `ds_idx`. Never looks at tokens. |
| [`OmniEnvironMeter.step(delta_time, global_step, module_metrics)`](../../../veomni/utils/omni_helper.py#L137) | [`OmniStepMetricsCallback.on_step_end`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L73) | [One DP all-reduce](../../../veomni/utils/omni_helper.py#L169) of FLOPs, sample count and per-module token sums, then the metrics listed below. |

The `OmniEnvironMeter` itself is built in
[`OmniStepMetricsCallback.__init__`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L51)
and stored as `trainer.environ_meter`.

## Lifecycle of one training step

- [`OmniTrainer.train_step`](../../../veomni/trainer/omni/omni_trainer.py#L512)
  - [`on_step_begin`](../../../veomni/trainer/omni/omni_trainer.py#L519) →
    [`OmniStepMetricsCallback.on_step_begin`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L64)
    - [`OmniEnvironMeter.add(micro_batch)`](../../../veomni/utils/omni_helper.py#L123) for each micro-batch: sample count, `ds_idx`.
    - [`start_time = time.time()`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L67)
  - For each micro-batch: [`forward_backward_step`](../../../veomni/trainer/omni/omni_trainer.py#L470) →
    [`OmniModel.forward`](../../../veomni/models/seed_omni/modeling_omni.py#L461) →
    [`TrainNodeRunner`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L78) →
    [`execute_train_node`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L27), for each node:
    - [`raw.pre_forward(method, **batch)`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L63)
      - Module code: `self.metric_meter_set_seqlens(method, full_lengths)`, before the SP slice.
    - [`raw.metric_meter_add(method)`](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L66): moves the stash for this `method` into the step buffer and clears the rest.
    - [Endpoint](../../../veomni/models/seed_omni/accelerated/utils/executor.py#L68), then `post_forward`.
  - [`clip_grad_norm` / `optimizer.step` / `lr_scheduler.step`](../../../veomni/trainer/omni/omni_trainer.py#L533)
  - [`on_step_end`](../../../veomni/trainer/omni/omni_trainer.py#L538) →
    [`OmniStepMetricsCallback.on_step_end`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L69)
    - [`delta_time = time.time() - start_time`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L72)
    - [`OmniModelRuntime.metric_meter_collect()`](../../../veomni/models/seed_omni/accelerated/omni_model/omni_model_runtime.py#L546) →
      per metered module [`(estimate_flops(buffer), buffer)`](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L150); buffer reset.
    - [`OmniEnvironMeter.step(delta_time, global_step, module_metrics)`](../../../veomni/utils/omni_helper.py#L137)
    - [`trainer.step_env_metrics = ...`](../../../veomni/trainer/callbacks/omni_callbacks/step_metrics_callback.py#L107),
      logged by [`WandbTraceCallback`](../../../veomni/trainer/callbacks/trace_callback.py#L157).

`delta_time` covers every micro-batch's forward and backward plus the optimizer
step. Metering only happens through `TrainNodeRunner`: calling
`OmniModel.forward` without a `node_runner` (the eager, framework-free path,
[`OmniModel._run_train_node`](../../../veomni/models/seed_omni/modeling_omni.py#L440))
runs `pre_forward` but never drains the stash, so nothing is counted there.

## Resulting metrics

[`OmniEnvironMeter.step`](../../../veomni/utils/omni_helper.py#L137) returns
these keys; `OmniStepMetricsCallback` merges the `training/*` losses into the
same dict and `WandbTraceCallback` logs it.

| Key | Meaning |
|-----|---------|
| [`flops_achieved(T)`](../../../veomni/utils/omni_helper.py#L177) | Sum over modules and DP ranks of `estimate_flops`, divided by `delta_time`. |
| [`flops_promised(T)`](../../../veomni/utils/omni_helper.py#L178) | Device peak TFLOPs ([`get_device_flops`](../../../veomni/utils/count_flops.py#L26)) × world size. |
| [`mfu`](../../../veomni/utils/omni_helper.py#L179) | `flops_achieved / flops_promised`. |
| [`consumed_chunk_num`](../../../veomni/utils/omni_helper.py#L183) | Cumulative real training samples (conversations) across DP ranks. |
| [`trace/<module>/tokens_per_second(M)`](../../../veomni/utils/omni_helper.py#L193) | This module's global tokens this step / `delta_time`. |
| [`trace/<module>/consume_tokens(M)`, `(B)`](../../../veomni/utils/omni_helper.py#L194) | This module's cumulative global tokens. |
| [`trace/<module>/avg_seq_len`](../../../veomni/utils/omni_helper.py#L197) | This module's global tokens this step / `global_batch_size`. |
| `max_memory_allocated(GB)`, `cpu_used_memory(GB)`, ... | Device and host memory, reduced to the worst rank by [`compute_device_memory_metrics`](../../../veomni/utils/helper.py#L162) (shared with `EnvironMeter`). |
| `multi_source/...` | Only with `data.enable_multisource`; see [Multi-source accounting](#multi-source-accounting). |

`<module>` is the module's name in the `OmniModel` (the key of
`OmniModelRuntime.module_runtimes`). While no module is metered, the three FLOPs
keys are [omitted](../../../veomni/utils/omni_helper.py#L185) rather than logged
as zero; token and memory metrics still appear.

FLOPs are summed across modules because each module's compute is distinct
(backbone layers vs. ViT vs. `lm_head`). Tokens are **not** summed: the
backbone's sequence already contains the text and image tokens the encoder
modules also count, so a merged total would double-count. That is why tokens are
reported per module under `trace/<module>/` and there is no top-level
`tokens_per_second(M)`.

`consume_tokens` is part of
[`OmniEnvironMeter.state_dict()`](../../../veomni/utils/omni_helper.py#L111),
which `GlobalStateCallback`
[saves](../../../veomni/trainer/callbacks/global_state_callback.py#L138) and
[restores](../../../veomni/trainer/callbacks/global_state_callback.py#L288), so
the cumulative counters survive a resume.

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
- **One FLOPs formula per module.** The step buffer does not record which
  call-site a length came from, so `estimate_flops` prices every length the
  same. Every invocation is counted: a module that appears at several nodes of
  a graph, or runs on every micro-batch, adds its lengths each time, and
  `metric_meter_collect` prices them all once at step end. Stash at each
  call-site only when they cost the same per token, or never run in the same
  graph (such as `encode` and `pack_encode`). When per-token costs differ,
  stash at one call-site and fold the others' cost into the formula: the text
  encoder counts on `encode` and prices `lm_head` there, stashing nothing on
  `decode`.
- **`estimate_flops` covers forward + backward and returns TFLOPs.** It must
  return `0.0` for `[]`, because a rank whose batch skipped the module still
  collects it. Count only parameters the module owns; do not reuse
  `VeomniFlopsCounter`'s whole-model formula.
- **Stash in every `pre_forward` that should count.** The stash is
  [cleared](../../../veomni/models/seed_omni/mixins/metric_meter_mixin.py#L137)
  right after each `pre_forward`, so re-reading it cannot double-count, a module
  invoked by several nodes is counted once per invocation, and a `pre_forward`
  that returns without stashing counts nothing rather than an earlier node's
  lengths.
- **Count the compute the module actually does.** Frozen modules are drained
  like any other metered module. If no gradient flows through a frozen module,
  it only runs a forward, and its `estimate_flops` should not include a
  backward term.

### Multi-source accounting

With `data.enable_multisource`, the multi-source tracker needs one sequence
length per sample to attribute tokens to each dataset.
[`OmniEnvironMeter._multisource_module`](../../../veomni/utils/omni_helper.py#L229)
picks the source module collectively: among the modules whose step buffer has
exactly one entry per sample (`len(seqlens) == len(ds_idx)`) on **every** DP
rank, it takes the one with the most global tokens, normally the backbone, whose
sequence is the union of all modalities. Ties resolve by name. If no module
qualifies, multi-source token counts are
[logged as zero](../../../veomni/utils/omni_helper.py#L207) with a one-time
warning. To make a backbone eligible, stash exactly one length per conversation.

## Tests

- [`tests/seed_omni/mixins/test_metric_meter_mixin.py`](../../../tests/seed_omni/mixins/test_metric_meter_mixin.py):
  the mixin's stash/accumulate/reset contract and the `OmniEnvironMeter` roll-up,
  including a two-rank gloo run.
- [`test_metric_meter_collect_drains_only_the_metered_modules`](../../../tests/seed_omni/runtime/test_omni_model_runtime.py#L318):
  the runtime drain.
- [`tests/seed_omni/trainer/test_step_metrics_callback.py`](../../../tests/seed_omni/trainer/test_step_metrics_callback.py):
  the callback wiring.
