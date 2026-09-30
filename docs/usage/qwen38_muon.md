# Qwen3.8 mixed Muon/AdamW candidate

`train.optimizer.type: qwen38_muon` is an opt-in correctness-first optimizer
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
- Indexer, vision tower and CPU PLE tables must remain frozen. Unknown trainable
  maps and LoRA parameters fail closed rather than silently receiving Muon.

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

The first version reconstructs one parameter's matrix shards at a time,
preserving whole-expert ownership on the expert axis. NS processes one logical
matrix at a time. It never orthogonalizes an FSDP fragment or combines experts
into one matrix. Replicated ranks currently repeat the NS calculation; balanced
matrix owners and asynchronous offload are future performance work.

An explicit reconstruction size limit is checked before allocation. The limit
bounds tensor sizes, not the complete peak memory: collective temporaries and
allocator reservations still require native measurement. CPU-offloaded Muon
parameters temporarily retain a zero second-moment tensor for compatibility
with the existing exact-dtype DCP schema. No CPU memory saving from removing
this unused state is claimed.

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
