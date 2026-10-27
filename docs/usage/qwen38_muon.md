# Qwen3.8 mixed Muon/AdamW candidate

`model.optimizer.type: qwen38_muon` is an opt-in correctness-first optimizer
for the Qwen4-Exp implementation of Qwen3.8. It leaves model forward/backward,
Ulysses/CP, expert routing, and global gradient clipping unchanged. The generic
`muon` optimizer and existing AdamW recipes retain their existing behavior.

## Recipe

- FP32 master weights and FP32 optimizer states; BF16 model compute copies.
- Muon: Nesterov momentum 0.95, eight explicitly pinned Polar Express iterations,
  Frobenius normalization epsilon `1e-14`, update scale `0.2*sqrt(max(rows, cols))`.
  Hardware HF32/TF32 down-conversion and autocast are disabled during NS.
- QSA Q/K/V and GDN Q/K/V input maps are orthogonalized separately per head.
  The interleaved QSA output-gate rows use AdamW within the same Parameter.
- Expert gate/up halves are separate matrices, independently per expert.
  Output projections and PLE key/value projections are whole linear maps.
- Embeddings, LM head, router, gated-residual projections, GDN z/a/b,
  normalization, convolution, and bias parameters use AdamW.
- Trainable PLE lookup tables use Adam (the AdamW fallback with zero weight
  decay). Include `ngram_embedding` in `no_decay_params`; nonzero decay on these
  tables is rejected. Their FP32 optimizer states can use the existing selective
  CPU offload while the compute tables retain their distributed device placement.
- Indexer and vision tower must remain frozen. The optional CPU-only PLE lookup
  remains a frozen-table feature; disable it to train the lookup tables. Unknown
  trainable maps and LoRA parameters fail closed rather than silently receiving Muon.

The schedule is generated from the Polar Express authors' implementation with
`l=1e-3, num_iters=8, degree=5, safety_factor_eps=1e-2, cushion=0.02` and pinned in
source. The Qwen report does not provide its numerical coefficient table, so
this is **not** a claim of bitwise equivalence to unpublished Qwen training code.
See [Qwen report section 3.1](https://arxiv.org/html/2608.30320v1#S3.SS1) and
[Polar Express](https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py).

This profile fixes the above Muon knobs; generic `muon_*` tuning fields belong
to `type: muon` and do not change this recipe. `lr`, `betas`, weight decay and
scheduler fields still apply. Muon and AdamW share the configured SFT LR; the
recipe does not import the pretraining LR from the paper.

## Offload and distributed semantics

Set `cpu_offload_param_patterns` explicitly, normally to the routed experts.
Both `cpu_offload_resident_moment_dtype` and `cpu_offload_cpu_moment_dtype` must
be `float32`. Only optimizer states are offloaded; model weights and gradients
retain their existing placement.

Native FSDP CPU-offloaded Muon weights need a runtime that explicitly registers
the accelerator compute device on those parameters. The public runtime does
not yet provide that integration and rejects the combination; optimizer-state
CPU offload above remains supported.

The first version reconstructs one parameter's matrix shards at a time,
preserving whole-expert ownership on the expert axis. NS processes one logical
matrix at a time. It never orthogonalizes an FSDP fragment or combines experts
into one matrix. Replicated ranks currently repeat the NS calculation; balanced
matrix owners and asynchronous offload are future performance work.

An explicit reconstruction size limit is checked before allocation. The limit
bounds tensor sizes, not the complete peak memory: collective temporaries and
allocator reservations still require native measurement. Pure three-dimensional
Muon expert parameters omit their unused second-moment tensor; mixed Muon/Adam
parameters retain the second moment used by the Adam slices.

### Optional local-disk PLE moments

For trainable, CPU-offloaded PLE lookup tables, set
`model.optimizer.qwen38_ple_moments_scratch_dir` to an existing, private local
`ext4`, `xfs`, or `btrfs` directory and explicitly set
`model.optimizer.qwen38_ple_moments_world_profile: true`. Every world rank must
use this same opt-in, directory path, optimizer partitions and step order; this
profile does not support an optimizer confined to a world subgroup. All local
ranks must share the same directory so disk reservations are coordinated.
Unresolved overlay, network, and memory-backed filesystems are rejected. Both
options are disabled by default. Ordinary optimizers do not use file-backed
state; standard trainer startup still performs a WORLD negotiation to detect
whether any rank selected this profile. Every WORLD rank must enter standard
trainer startup together, including for non-file optimizers.

The lookup tables' FP32 first and second moments use unnamed local files;
master weights, model compute precision, and optimizer arithmetic are unchanged.
The trainer cursor adds a resume-identity field. Reserve local disk for both moments for every local
shard, plus at least 64 GiB free for other writers. This allocation uses real
filesystem blocks, not sparse files. A controlled flush and page eviction occurs
after each optimizer chunk and DCP read item, but the operating system's page
cache, checkpoint read buffers, and other model states still count toward host
memory. Measure the full profile before relying on a memory budget.
During DCP restore, successfully consumed source ranges receive a read-only
`POSIX_FADV_DONTNEED` hint. Kernel/filesystem reclamation is not guaranteed,
and concurrent readers may need to fetch those bytes again; checkpoint data
and FP32 values are not modified.

Scratch files are not checkpoints and disappear when the process exits. On an
I/O failure, discard the failed process and start again from a completed DCP;
do not continue with a partially updated optimizer. This mode requires
`train.checkpoint.save_async: false`; synchronous optimizer checkpoint writes
use one writer thread per rank to surface I/O errors. Tensor payloads are
grouped into 256 MiB DCP files to avoid retaining an entire rank's device-to-host
copies. An individual tensor item can exceed that bound; opaque metadata,
allocator overhead and other model state also contribute to peak host memory.
The additional files and serial writes may increase save time. A per-module
exclusive claim prevents cooperating writers from saving to
the same destination concurrently; stale claims from an interrupted process
need operator inspection. Concurrent unrelated writers must use distinct paths.
After an interrupted save, select a new output root when resuming from the last
complete checkpoint. This profile refuses to overwrite a partial or complete
checkpoint at the old step; automatic same-path retry is not supported.
Specify both paths explicitly when switching output roots:

```yaml
train:
  checkpoint:
    output_dir: /new/run
    load_path: /old/run/checkpoints/global_step_100  # last completed step
```

Do not use `load_path: auto` for this first restart: it searches the new output
root and can otherwise start from step zero.
Disk I/O can also increase optimizer-step time.

Exact trainer resume in this profile requires the data provider to attach an
explicit exact-checkpoint contract to its loader. The contract installs its
own supported snapshot implementation and validates saved cursors and
checkpoint boundaries; unknown providers are rejected. Each provider must
document and test its own worker, replay and epoch-boundary restrictions.
This file-backed profile currently requires a multisource SFT meter; single
source and preference/DPO data are outside its validated scope.
New checkpoints bind data, parallel and batch configuration, model semantics,
world size, epoch count, learning-rate schedule, optimizer and clipping
configuration to the cursor. A changed schedule or an older checkpoint
without this identity requires an explicit migration, since PyTorch does not
serialize scheduler closures. Scratch paths can change on restart. The identity
does not prove that mutable dataset contents or tokenizer assets at an
unchanged path are identical; immutable input snapshots must be supplied by
the caller. Epoch-boundary resume is not yet certified in this profile.
The controlled launcher must validate its actual image, framework, trainer,
entrypoint and overlay identities, and set `VEOMNI_EXACT_RESUME_RUNTIME_ID` to
their combined `sha256:<64 lowercase hex>` fingerprint. The checkpoint binds
that declaration; the environment variable itself cannot prove the running
bytes. The same exact identity must be supplied on restart. In multisource
mode the saved sampler step and accumulated counts are checked before any
cursor is applied. Checkpoints after early data exhaustion or at an epoch
boundary fail closed until their replay behavior is validated.
Deploy an immutable source checkout or image from before Python imports until
the process exits: runtime source hashes detect later changes but cannot
prove which bytes were already imported before installation.

The provider's `_veomni_exact_checkpoint_contract` supplies `install()`,
`validate_cursor(trainer, cursor, require_cursor=...)` and
`validate_before_checkpoint(trainer, state)`, and a stable
`checkpoint_identity()` string available after installation. These callbacks
run on every participating rank; the trainer checks the identity across ranks
and at recovery. Validation
raises locally and the trainer coordinates
errors. The provider is responsible for ensuring that a saved cursor captures
its own prefetch, worker and replay state and that fresh-process restoration
returns the same logical samples.

## Checkpoint contract and validation

Each parameter has one optimizer owner and a serialized logical-slice plan.
Source DCP metadata must contain the full optimizer recipe and state; partial
AdamW-to-Muon resume is rejected before placeholder values can hide missing
fields. A fresh Muon run starts from model weights with new optimizer state.
Existing AdamW moments are never silently reinterpreted as Muon momentum.

Local tests cover FP64 polynomial/update references, BF16 compute copies with
FP32 masters, row/column/whole-expert sharding, exact recipe rejection and DCP
restore. These are not a substitute for native NPU numerical, memory,
throughput, and fresh-process checkpoint validation. Keep the profile a
candidate until those gates pass.


Trainable PLE checkpointing keeps each table owner's lazy Adam counter on an
explicit owner-grid DTensor. Owners that have not received lookup requests
contribute zero-moment, zero-step buffers to the snapshot so DCP sees complete
coverage. This does not run an optimizer update or initialize live save-side
state. On resume, zero-step owners become lazy again and create their master
from the restored model on their first real update. Owner-grid identity is
versioned; old scalar-counter PLE checkpoints or changed mesh rank layouts
require explicit migration. This does not change the checkpoint layout of
frozen PLE or non-PLE Muon parameters.
