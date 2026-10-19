# FP32 master weights and optimizer-only CPU offload

Opt in with optimizer type `master_fp32_adamw`, `full_fp32_adamw` or `cpu_offload_adamw`. Existing AdamW remains unchanged. Master weights and update arithmetic are FP32; compute weights may remain BF16. BF16 moment storage is lossy and is not equivalent to FP32 moments.

For CPU offload, `cpu_offload_param_patterns` selects parameters by full parameter-name regex. `cpu_offload_cpu_moment_dtype` and `cpu_offload_resident_moment_dtype` independently choose `float32` or `bfloat16`; both default to FP32. Run optimizer steps after gradient synchronization and global clipping. Parameters and gradients stay on the accelerator; selected master/moment states are stored and updated on CPU using bounded staging.

DCP load preserves offloaded CPU storage instead of temporarily materializing the optimizer on the accelerator. Precision-sensitive restores reject dtype changes and partial optimizer fields. A parameter with no saved state at all keeps the fresh optimizer's default lazy state; parameters that never received a gradient can legitimately be absent. A zero-step sparse `MultiOptimizer` checkpoint carries an exclusive, transactional marker and restores the target to lazy state. A marker combined with tensor state is rejected rather than overwriting the load, and marker-less checkpoints with no optimizer state remain rejected because they are indistinguishable from lost state.

CPU tests cover FP32 reference updates, cold optimizer-state restoration, missing-field rejection and precision migration rejection. Accelerator transfers, DTensor layouts and multi-rank DCP cold resume still require native qualification on each hardware profile.

Resume validates saved tensor shapes and dtypes before DCP reads. Precision-sensitive optimizers must expose complete target state for every trainable parameter; use a fresh optimizer when resuming rather than one initialized for only a subset of parameters. Partial ordinary AdamW state remains supported. These checks allocate no additional master or moment tensors.
