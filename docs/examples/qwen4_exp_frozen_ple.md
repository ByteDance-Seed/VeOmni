# Freeze Qwen4-Exp PLE embeddings

To freeze the per-layer embedding (PLE) n-gram tables, merge this YAML fragment
into your Qwen4-Exp training configuration.

```yaml
train:
  freeze_ple_embeddings: true
  frozen_ple_dtype: bfloat16
```

`train.freeze_ple_embeddings` defaults to `false`. It applies only to models
with `model_type: qwen4_exp` that have
`ple.ple_embedding.ngram_embedding.shard_<n>.weight` parameters. The forward
pass still looks up values in these tables. Freezing disables gradients for
the table weights and excludes them from optimizer parameter groups and Adam
state. Other PLE parameters retain their existing training behavior.

`train.frozen_ple_dtype` accepts `float32` or `bfloat16` and requires
`freeze_ple_embeddings: true`. It sets the in-memory dtype of the frozen
PLE table weights during training. If omitted or set to `null`, the weights
keep the dtype they have after model parallelization.

## Distributed training

- **CPU parameter offload.** Each rank's frozen PLE table shards must reside
  on the training device when VeOmni checks their placement. Offloading
  these shards to CPU fails that check.
- **DDP.** The current implementation does not support `fsdp_mode: ddp`
  with `freeze_ple_embeddings: true`. The model type check reads `model.config`,
  while the DDP wrapper stores the original model at `model.module`.
  `fsdp_mode` defaults to `fsdp2`, which is also the setting in the Qwen4-Exp
  training example.

## Resume from a checkpoint

- **Frozen tables.** VeOmni freezes the tables before restoring the checkpoint.
  Loading weights into the tables preserves their configured training dtype and
  frozen status.
- **Optimizer state.** If the PLE tables were trainable when the checkpoint
  was saved, the saved optimizer parameter groups may include them.
  Resuming with `freeze_ple_embeddings: true` excludes them from the current
  optimizer. This mismatch can cause optimizer state loading to fail.
