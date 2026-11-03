# File-backed FP32 PLE optimizer moments

This is an opt-in memory-control path for the Qwen3.8 Muon/AdamW optimizer.
PLE lookup master weights and both Adam moments remain FP32. The moments use
private local disk-backed mappings; they are reconstructed from DCP on resume.
This does not quantize optimizer state or change the forward compute precision.

Set `qwen38_ple_moments_world_profile: true` and
`qwen38_ple_moments_scratch_dir` to the same absolute path on every rank. The
path must exist, be owned by the training user with no group/other permissions,
and reside on an ext4, XFS or Btrfs local disk. Each allocation reserves a
further 64 GiB of free disk. Network and memory-backed filesystems are refused.
Use a dedicated scratch path and monitor disk space and I/O.

All ranks in WORLD must participate with matching logical parameter groups,
optimizer hyperparameters, scratch profile and training geometry throughout
training, including scheduler updates. Per-rank changes to optimizer groups,
hooks or hyperparameters are unsupported and may desynchronize collectives;
they are not checked by a WORLD agreement on every step. The PLE lookup must
use CPU optimizer offload. Synchronous DCP is required; asynchronous saves,
closures and optimizer step hooks are rejected. An I/O failure poisons the
optimizer and requires recovery from a complete checkpoint in a fresh process.
Scratch files are temporary and are not themselves checkpoints.

Checkpoint provenance identifies the file-backed profile. Changing an existing
checkpoint between regular and file-backed state, changing an EP owner layout,
or resuming an older checkpoint without sufficient group/scheduler identity
requires a separately verified migration; normal resume fails closed. This
profile is not enabled by default and does not extend to LoRA or unrelated
optimizers. Its local memory and throughput impact must be measured on the
actual model, geometry and hardware; the CPU unit tests do not establish that a
particular full-model memory limit is met.
