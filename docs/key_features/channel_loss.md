# Channel Loss: Per-Source Causal-LM Observability

## Background

A single training loss can hide a deteriorating data source when other sources
improve or their mixture proportions change. Channel Loss reports detached
cross-entropy (CE) and supervised-token counts for each source in sampled
optimizer steps. It does not change the model's returned loss, data sampling
weights or gradients.

Read [data packing](../usage/data_packing_and_dyn_bsz.md) and
[trainer usage](../usage/trainer.md) when integrating a custom data pipeline.

## Features

### Loss attribution by data source

Suppose a training step mixes books and code. The overall loss gives one result
for the mixture. Channel Loss additionally asks: how well is the model predicting
the books, and how well is it predicting the code?

This requires two things: cross-entropy (CE, a measure of prediction error) for
each valid supervised token, and the source of the sequence containing that token.
The flow is:

```text
Keep each sequence's source → compute CE for valid supervised tokens
                           → sum loss and token counts per source → emit metrics
```

Packing puts multiple original sequences into one tensor row. If a row contains
"book sequence + code sequence", both source identities and their boundaries must
survive; attributing the whole row to books would be wrong. The callback saves
this alignment information before removing source fields the model does not need.

### Metric definitions and calculation example

The following is a calculation example, not a measured training result. Suppose
a sampled step successfully records:

| Source | Valid supervised tokens | CE sum | Mean CE | Weighted contribution |
| --- | --- | --- | --- | --- |
| Books A | 30 | 60 | `60 / 30 = 2` | `60 / 40 = 1.5` |
| Code B | 10 | 30 | `30 / 10 = 3` | `30 / 40 = 0.75` |

* **Mean CE (`channel_loss`)** asks for the average loss per valid token of this
  source. Code has higher mean CE than books in this example.
* **Weighted contribution (`channel_loss_weighted`)** asks how much this source
  contributes to the overall mean CE of the recorded data. Both sources use 40
  as the denominator, rather than their individual token counts.
* **Token count (`channel_tokens`)** shows how many valid supervised tokens support
  the metric, helping explain changes in source proportions and coverage.

In general, let `S_c` be source c's loss sum, `N_c` its valid token count, and
`N = sum_c N_c` the total successfully recorded count. These metrics are
`S_c / N_c`, `S_c / N` and `N_c`, respectively. A source with zero valid tokens
emits no metrics.

The example contributions sum to `1.5 + 0.75 = 2.25`, the overall mean CE of the
successfully recorded tokens. This need not equal the trainer's returned loss:
the objective may include auxiliary losses or DPO preference loss, or use a
different normalization. Skipped metadata can also make observation incomplete.

### Sampling cadence and gradient isolation

`interval: 10` samples optimizer steps 10, 20, 30, and so on. Each report covers
**all observed micro-batches in that step**, not the preceding ten steps. An
optimizer step may accumulate gradients over multiple micro-batches, so their
source totals must be combined before computing the means.

Observation uses detached tensors: these extra calculations do not contribute
gradients and do not rewrite the original objective. If the model provides logits
(scores used to form token prediction distributions), CE is computed from them.
A fused-loss path may not provide full logits; the observer then recomputes the
LM-head projection in chunks from hidden states and weights, converting hidden
representations to vocabulary scores. This explains why observation still adds
computation.

### Label alignment and parallel aggregation

A causal language model predicts the next token from the current position, so CE
must use the corresponding next-token label. Outside SP, labels are shifted unless
`shift_labels` is already supplied. Under sequence parallelism (SP), the collator
has already shifted them; they must not be shifted twice. Ignored labels and
masked padding do not contribute to the denominator.

SP splits sequences across processes, so one original sequence may cross a slice
boundary. The computer uses segment information to preserve its source, combines
statistics within SP where needed, and the callback sums each source's loss and
token counts across data-parallel (DP) processes before dividing. This produces
a valid-token-weighted mean, not a simple average of per-process mean losses.

## Configuration

All fields below are under `train.channel_loss`.

| Field | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | Enable the callback's observation path. |
| `interval` | `10` | Sample every N optimizer steps; must be at least 1. |
| `source_id_keys` | `[channel_id, source_id, dataset_id, ds_idx]` | Ordered candidate ID keys; the first present key is used. |
| `source_name_keys` | `[channel_name, source_name, dataset_name, data_name]` | Ordered optional display-name keys. |
| `extra_strip_keys` | `[cur_token_num]` | Additional keys removed before model forward. |
| `loss_metric_prefix` | `channel_loss` | Mean-CE metric prefix. |
| `weighted_loss_metric_prefix` | `channel_loss_weighted` | Weighted-contribution prefix. |
| `token_count_metric_prefix` | `channel_tokens` | Supervised-token prefix. |
| `log_weighted_loss` | `true` | Emit weighted contributions. |
| `log_token_count` | `true` | Emit token counts. |
| `strict` | `false` | Raise for missing source IDs or incompatible alignment instead of skipping invalid metadata. |

ID, name and extra-strip keys are removed while the feature is enabled, including
unsampled steps. Strict mode is a metadata contract check; it does not turn every
possible numerical/CE failure into an exception. Inspect warnings as well.

## Usage and metadata example

Merge this **configuration fragment** into a working causal-LM text or VLM recipe:

```yaml
train:
  channel_loss:
    enable: true
    interval: 10
    strict: true
    source_id_keys: [ds_idx]
    source_name_keys: [source_name]
    log_weighted_loss: true
    log_token_count: true
```

The native multi-source dataset attaches `ds_idx` and optional `source_name` to
samples. Custom pipelines must preserve equivalent metadata through transforms
and collation. An ID identifies a source; an optional name labels it. The ordered
ID/name lists must align with the original sequences packed into the batch,
not with every token and not just with the number of packed tensor rows.

For example, this is an **illustrative packed batch**, not a standalone model
forward (token IDs, dtype/device and other model-specific inputs are omitted):

```python
packed_metadata = {
    "position_ids": [[0, 1, 2, 0, 1]],
    "ds_idx": [0, 1],
    "source_name": ["books", "code"],
}
# Two original sequences: lengths 3 and 2, in the same source order.
# The real collator must supply tensor position_ids and masked causal labels.
```

Use the original recipe's launch command after merging. First use `interval: 1`
for a short diagnostic run and check metadata errors and token counts; then
increase the interval for normal training. The callback adds metrics to both
`step_train_metrics` and `step_env_metrics` for the existing trainer logging
callbacks. Keys use `<stable-source-id>__<sanitized-display-name>`, such as
`channel_loss/source-i-0__books` for integer source 0 named books. IDs prevent distinct
sources with colliding sanitized names from silently sharing a time series.