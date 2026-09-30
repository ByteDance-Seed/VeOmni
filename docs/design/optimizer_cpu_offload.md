# FP32 master weights and optimizer-only CPU offload

Opt in with optimizer type `master_fp32_adamw`, `full_fp32_adamw` or `cpu_offload_adamw`. Existing AdamW remains unchanged. Master weights and update arithmetic are FP32; compute weights may remain BF16. BF16 moment storage is lossy and is not equivalent to FP32 moments.

For CPU offload, `cpu_offload_param_patterns` selects parameters by full parameter-name regex. `cpu_offload_cpu_moment_dtype` and `cpu_offload_resident_moment_dtype` independently choose `float32` or `bfloat16`; both default to FP32. Run optimizer steps after gradient synchronization and global clipping. Parameters and gradients stay on the accelerator; selected master/moment states are stored and updated on CPU using bounded staging.

DCP load preserves offloaded CPU storage instead of temporarily materializing the optimizer on the accelerator. Precision-sensitive restores reject dtype changes and incomplete optimizer fields, including wholly absent lazy states for unused parameters. Such checkpoints need an explicit migration; silent initialization from pre-load compute weights is unsafe.

CPU tests cover FP32 reference updates, cold optimizer-state restoration, missing-field rejection and precision migration rejection. Accelerator transfers, DTensor layouts and multi-rank DCP cold resume still require native qualification on each hardware profile.
