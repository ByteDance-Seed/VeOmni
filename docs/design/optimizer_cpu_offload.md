# FP32 master weights and optimizer-only CPU offload

Opt in with optimizer type `master_fp32_adamw`, `full_fp32_adamw` or `cpu_offload_adamw`. Existing AdamW remains unchanged. Master weights and update arithmetic are FP32; compute weights may remain BF16. BF16 moment storage is lossy and is not equivalent to FP32 moments.

In a mixed optimizer, strict checkpoint dtype and shape checks apply to the parameters owned by precision-sensitive optimizer leaves. Unknown saved fields on those parameters require an explicit migration; ordinary AdamW leaves retain their existing loading policy.

Checkpointing parameters that remain owned by a precision-sensitive optimizer
while frozen, followed by unfreezing, is not yet supported safely. The save
rejects frozen parameters with existing optimizer history rather than silently
discarding it; restore also rejects frozen parameters still owned by such an
optimizer. Persistent PLE parameters may be rejected even without history.
Previously accumulated history for parameters omitted from a rebuilt target
optimizer also requires explicit migration if those parameters are later
unfrozen. Do not use these freeze/unfreeze transitions across a checkpoint
until a verified migration is available.

Use the framework `DistributedCheckpointer.load` for resume: it clears temporary optimizer handoff markers if DCP loading fails. Calling `torch.distributed.checkpoint.load` directly bypasses that cleanup if an error occurs after planning. Saving a single empty optimizer, or an ExtraParallel-aware `MultiOptimizer` containing an empty child, is not yet safe: PyTorch can materialize synthetic optimizer state while collecting it. Avoid those save paths until they are fixed.

The frozen-parameter checks below apply to `DistributedCheckpointer.save` and
`DistributedCheckpointer.load`; direct `dcp.save`/`dcp.load` using
`OptimizerState` bypass them and must not be used for freeze/unfreeze resumes.

For a single non-EP `cpu_offload_adamw`, `DistributedCheckpointer.load` also reads
the saved group names before model tensors and again after model load hooks,
before optimizer tensors. Legacy fused checkpoints are read in two phases for
precision-sensitive optimizers. The check rejects a
changed group membership or order, because a different offload selection can
change optimizer and scheduler groups; those changes require an explicit
migration. This check requires an immutable checkpoint during both reads. It
does not cover direct `dcp.load` or `MultiOptimizer`/EP. The checked single
optimizer path rejects FSDP1 flat parameters and custom optimizer state-dict
hooks or optimizer implementations. It also rejects step hooks on an uninitialized optimizer because PyTorch
may invoke them during synthetic optimizer-state initialization. A regular
load pre-hook still runs after the
preflight and is responsible for any changes it makes. The preflight does not
make later tensor I/O transactional: a failure can leave the model or optimizer
partially modified, so reconstruct them before another load attempt. In the
precision-sensitive path, rank-local model state hook failures are propagated
before peers continue; failures inside a model's own collective are not covered.

For CPU offload, `cpu_offload_param_patterns` selects parameters by full parameter-name regex. `cpu_offload_cpu_moment_dtype` and `cpu_offload_resident_moment_dtype` independently choose `float32` or `bfloat16`; both default to FP32. Run optimizer steps after gradient synchronization and global clipping. Parameters and gradients stay on the accelerator; selected master/moment states are stored and updated on CPU using bounded staging.

DCP load preserves offloaded CPU storage instead of temporarily materializing the optimizer on the accelerator. Precision-sensitive restores reject dtype changes and incomplete optimizer fields, including partially missing states for unused parameters. Such checkpoints need an explicit migration; silent initialization from pre-load compute weights is unsafe.

Optimizer load pre-hooks receive the complete saved state. Post-hooks run only after validated tensors are installed, so their moment changes survive into the next update without casting FP32 states through the compute dtype.

CPU tests cover FP32 reference updates, cold optimizer-state restoration, missing-field rejection and precision migration rejection. Accelerator transfers, DTensor layouts and multi-rank DCP cold resume still require native qualification on each hardware profile.

Resume validates saved tensor shapes and dtypes before DCP reads. Use a fresh optimizer when resuming. The loader prepares target placeholders, then removes those for parameters whose saved state is wholly absent. A partially initialized target optimizer is rejected if its existing state cannot represent the saved state. Partial ordinary AdamW state remains supported. Validation itself allocates no additional master or moment tensors.

Wholly absent lazy states are supported: DCP placeholders are removed on restore, and the optimizer initializes the FP32 master from the current parameter on its first real gradient. Partially missing fields or implicit precision migration remain errors. A zero-step non-ExtraParallel sparse `MultiOptimizer` checkpoint carries an exclusive marker and restores lazy state. A marker combined with tensor state is rejected; marker-less checkpoints with no optimizer state remain rejected because they are indistinguishable from lost state.

Same-dimension ExtraParallel/FSDP checkpoints carry an explicit EP-outer layout marker in model and optimizer state. Older unversioned layouts require an explicit migration and are rejected before tensor reads, even during partial/optimizer-only restore. The marker is not a model weight and is excluded from HF export.
