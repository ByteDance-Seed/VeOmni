# Freeze Qwen4-Exp PLE embeddings

Use these VLM-only options with `tasks/train_vlm.py` to freeze Qwen4-Exp
per-layer embedding (PLE) n-gram tables. The plain-text training entry point
rejects these options as unknown fields.

```yaml
train:
  freeze_ple_embeddings: true
  frozen_ple_dtype: bfloat16
```

`train.freeze_ple_embeddings` defaults to `false` and accepts YAML booleans
(`true` or `false`), not quoted strings. It applies only to models
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
- **DDP.** Qwen4-Exp configurations with ExtraParallel require
  `fsdp_mode: fsdp2`, the default and the setting in the Qwen4-Exp example.
  DDP does not apply the required sharding plan, independently of PLE freezing.

## Resume from a checkpoint

- **Frozen tables.** Freezing happens before parallelization; the requested dtype
  is applied afterwards, before optimizer construction. Checkpoint loading preserves
  the tables' configured dtype and frozen status.
- **Optimizer state.** If the PLE tables were trainable when the checkpoint
  was saved, the saved optimizer parameter groups may include them.
  Resuming with `freeze_ple_embeddings: true` excludes them from the current
  optimizer. This mismatch can cause optimizer state loading to fail.
