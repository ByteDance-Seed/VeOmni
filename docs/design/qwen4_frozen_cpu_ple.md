# Opt-in frozen CPU PLE storage

The model helper `veomni.models.transformers.qwen4_exp.cpu_ngram` moves only already-local, frozen vocabulary shards to CPU. It leaves projection/gating parameters and distributed request routing unchanged. It is not activated automatically.

Call `validate_cpu_ngram_config(args)` and `freeze_ngram_tables(model)` before parallelization; call `install_cpu_ngram(model)` after HF loading and FSDP wrapping, before optimizer construction. Only persistent two-dimensional, FSDP-ignored DTensors with `(Shard(1), Shard(0))` are accepted. On checkpoint restoration call `refresh_cpu_ngram_lookup(model)` after DCP has loaded the weights and before the next forward.

Only synchronous sharded DCP is supported; reject HF export, async save and torch.compile. The caller must wire these lifecycle points explicitly. CPU tests verify local lookup identity, order, duplicates, empty requests and input validation; they do not qualify distributed installation or DCP cold resume on an accelerator.
