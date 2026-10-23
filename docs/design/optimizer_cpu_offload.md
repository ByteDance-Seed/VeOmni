# FP32 master weights and optimizer-only CPU offload

Opt in with optimizer type `master_fp32_adamw`, `full_fp32_adamw` or `cpu_offload_adamw`. Existing AdamW remains unchanged. Master weights and update arithmetic are FP32; compute weights may remain BF16. BF16 moment storage is lossy and is not equivalent to FP32 moments.

For CPU offload, `cpu_offload_param_patterns` selects parameters by full parameter-name regex. `cpu_offload_cpu_moment_dtype` and `cpu_offload_resident_moment_dtype` independently choose `float32` or `bfloat16`; both default to FP32. Run optimizer steps after gradient synchronization and global clipping. Parameters and gradients stay on the accelerator; selected master/moment states are stored and updated on CPU using bounded staging.

DCP load preserves offloaded CPU storage instead of temporarily materializing the optimizer on the accelerator. Precision-sensitive restores reject dtype changes and incomplete optimizer fields, including partially missing states for unused parameters. Such checkpoints need an explicit migration; silent initialization from pre-load compute weights is unsafe.

CPU tests cover FP32 reference updates, cold optimizer-state restoration, missing-field rejection and precision migration rejection. Accelerator transfers, DTensor layouts and multi-rank DCP cold resume still require native qualification on each hardware profile.

Resume validates saved tensor shapes and dtypes before DCP reads. Use a fresh optimizer when resuming. The loader prepares target placeholders, then removes those for parameters whose saved state is wholly absent. A partially initialized target optimizer is rejected if its existing state cannot represent the saved state. Partial ordinary AdamW state remains supported. Validation itself allocates no additional master or moment tensors.

Wholly absent lazy states are supported: DCP placeholders are removed on restore, and the optimizer initializes the FP32 master from the current parameter on its first real gradient. Partially missing fields or implicit precision migration remain errors. A zero-step sparse `MultiOptimizer` checkpoint carries an exclusive marker and restores lazy state. A marker combined with tensor state is rejected; marker-less checkpoints with no optimizer state remain rejected because they are indistinguishable from lost state.

Same-dimension ExtraParallel/FSDP checkpoints carry an explicit EP-outer layout marker in model and optimizer state. Older unversioned layouts require an explicit migration and are rejected before tensor reads, even during partial/optimizer-only restore. The marker is not a model weight and is excluded from HF export.

CPU optimizer receipts use framework logging. DCP's synthetic zero-learning-rate state initialization does not consume the one-shot receipt; it is emitted once updates use a nonzero learning rate.
