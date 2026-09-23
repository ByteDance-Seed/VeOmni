# DiT Fixed Microbatches

`DiTTrainer` honors `train.micro_batch_size` instead of overriding it to one.
Existing recipes with `micro_batch_size: 1` are unchanged. A larger microbatch
does not automatically enable packing or guarantee a speedup: each model decides
whether to loop over samples, batch or pack them, or reject unsupported inputs.

## Training batch sizes

DiT keeps `train.dyn_bsz = false`. With microbatch size `M`, data-parallel size
`D`, and `K` gradient-accumulation microbatches per optimizer step:

```text
G = M × D × K

dataloader_batch_size = G / D = M × K
```

For example, this overlay uses `K = 2` when `D = 4`, provided the chosen model
supports two samples per microbatch:

```yaml
train:
  training_task: offline_training
  dyn_bsz: false
  micro_batch_size: 2
  global_batch_size: 16
```

`MakeMicroBatchCollator` splits each dataloader batch into `K` microbatches.
`DiTDataCollator` keeps each one as a `dict[str, list]`, with `M` entries per
column. It does not concatenate tokens or slice inputs for sequence parallelism.

The existing training path is unchanged:

```text
get_condition? → process_condition(**batch) → model(**batch) → outputs.loss / K
```

`online_training` runs `get_condition` first; `offline_training` reads cached
conditions. Both use the same condition-processing and model-forward interfaces.
Each scalar in `outputs.loss` must be a **sample mean** within the microbatch.
Models must average per-sample losses, not all packed tokens, so longer samples
do not receive more weight. The trainer then divides by `K` for accumulation.

## Offline embedding

`offline_embedding` runs only the condition model and writes one parquet row per
sample; it never packs or invokes the DiT. It also honors `M`, but overrides the
training global batch to `G = M × D`, with `dataloader_batch_size = M` and `K = 1`.
There is no optimizer, and keeping one microbatch per step limits the amount of
video data serialized by the sequence-parallel broadcast. Choose `M` to fit the
encoder and host-memory budget. Embedding disables sampler shuffling and pads
only an index view of the source dataset; the saver discards trailing repeats
so each original sample is written once.

## Model ownership

Packing, valid-token selection, multimodal layouts, attention boundaries,
positions and output reconstruction remain inside modeling. SP/CP slicing also
belongs there, after patchification where applicable, not in the DiT collator.
A model that cannot handle a multi-sample or parallel configuration must reject
it itself. The trainer adds no packing switch, alternate sample/output protocol,
or model-capability envelope.

This is fixed-sample batching, independent of the token-budget scheduling used
by dynamic batching. MiniMax H3 packing is a separate model implementation; this
trainer change alone does not lift H3's single-sample condition-processing check.
