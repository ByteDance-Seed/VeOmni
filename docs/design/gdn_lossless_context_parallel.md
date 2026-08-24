# Lossless GDN Context Parallelism

This document defines the production correctness foundation for Qwen3.5
GatedDeltaNet context parallelism. Both algorithms share one validated token
ownership, pad, BOS, and causal-convolution halo contract.

## Configuration and scope

```yaml
train:
  dyn_bsz: true
  accelerator:
    cp_size: 8
    ulysses_size: 4
model:
  ops_implementation:
    gdn_context_parallel_implementation: state_passing_lossless
```

`state_passing_lossless` sends the recurrent state between consecutive native
chunk owners. On Ascend, `kcp` keeps the same physical↔owned sparse-packed
all-to-all and halo route, but replaces recurrent-state P2P with an fp32 affine
summary all-gather and prefix composition. Its local pre-scan is the fixed TTX
BC8/M1 backend (forward column tile 32, backward time tile 128, replay column
tile 8); there are no environment-variable backend overrides or silent torch
fallbacks. GPU model paths reject `kcp` explicitly.

`headwise_lossless` is the communication-throughput candidate for hybrid
CP×Ulysses. Full-attention layers retain their physical Ring/Hybrid layout;
each GDN layer packs q/k/v/b/a into one equal-split `all_to_all_single`, gathers
the canonical packed sequence, and shards GDN heads over the flattened
CP-major/Ulysses-inner group. The inverse exchange restores physical tokens.
It uses the ordinary fused GDN forward/backward with no recurrent-state
approximation or duplicated recurrence. Padding is removed per sample before
GDN and restored as zero afterward.

The current CP release targets packed text training on Ascend NPU with a
power-of-two CP size. CPU execution is reserved for correctness oracles, and
CUDA CP is unsupported, including the generic Ring/Hybrid path. Planner and CPU
distributed-oracle coverage includes CP2/4/8/16. Full attention uses causal Ring
CP; GDN uses lossless chunk ownership plus state passing/KCP, or the explicit
headwise lossless route. Unsupported GDN
inputs include selector without CP, non-packed batches, attention dropout,
sliding-window/softcap, non-causal or multimodal/cross-attention, a missing
hardware-specific fused-attention backend, and `cp_size > 1` without an explicit
lossless GDN selector for Qwen3.5. The backwards-compatible `disabled` selector
disables only the GDN-specific state-passing/KCP algorithm. With `cp_size > 1`,
it selects generic Ring/Hybrid CP for non-GDN causal models and does not disable
CP itself; Qwen3.5 never silently falls back to it.

## Layout contract

1. Each packed sample is padded independently to `2 * CP * Ulysses`. The
   physical layout assigns the early and mirrored late causal chunks to a CP
   rank, then slices that shard contiguously across Ulysses ranks.
2. Full attention stays in the physical mirrored layout. Ring backward
   re-circulates `[K, V, dK, dV]`; it retains only owner-local K/V rather than a
   full global KV cache.
3. GDN derives ownership from the original valid sample lengths. A native
   64-token chunk belongs to exactly one rank; only the final sample tail may
   be partial. Owners are contiguous and monotonic.
4. A reversible variable-split all-to-all maps physical tokens to owners and
   back. Physical pad tokens are omitted from the wire and restored as zeros;
   their inverse gradient is zero.
5. The BOS owner starts from zero state. Each later active owner receives the
   previous active owner's final recurrent state. Causal-conv halo follows the
   same predecessor relation. Empty owners still participate in collective
   ordering and autograd dependencies.
6. KCP gathers one affine transform per sample and rank. BOS and inactive
   states remain numeric zero but retain the all-gather/reduce-scatter
   autograd dependency on every rank. The collective payload depends on
   `CP × H × K × (K+V)`, not sequence length.
7. Headwise lossless requires the flattened rank identity
   `sp_rank = cp_rank * ulysses_size + ulysses_rank`. It packs all five GDN
   projections into one sequence→head A2A and uses one inverse output A2A;
   rank-order or metadata disagreement fails before the payload collective.

All ranks compile the same plan and exchange its digest before the first
all-to-all. A topology, CU, split, route, rank, or hash mismatch aborts before
communication instead of risking a collective hang.

## Correctness contract

- ownership is deterministic for CP 1/2/4/8/16 and Ulysses 1/4;
- physical → owned → physical is token-exact, including empty samples/ranks,
  non-aligned tails, and per-sample padding;
- recurrent-state and halo P2P preserve cross-rank backward edges;
- Ring forward and `dq/dk/dv` match a dense packed causal oracle;
- unknown selectors/backends and unsupported hardware paths fail closed;
- KCP local affine summaries match the portable recurrence, and distributed
  CP2/4/8/16 prefix full gradients match a monolithic CPU oracle;
- headwise packed/GQA forward and full VJP match a monolithic oracle, including
  empty segments, hybrid Ulysses×CP rank order, and non-reentrant checkpointing;
- `gdn_cp_runtime_evidence.snapshot()` returns a public typed identity and
  A2A/P2P/AG enter-exit-error counters without log-marker parsing;
- generated Qwen3.5 and Qwen3.5-MoE GPU/NPU modeling files must pass
  `patchgen --check` and must never be edited directly.

The CPU reference attention backend exists only for unit tests. Long-sequence
production training in this release requires Ascend fusion attention; CUDA
FlashAttention CP is not supported. The Ring scheduling helpers are adapted
from MindSpeed under BSD-3-Clause; source files preserve Huawei attribution and
identify ByteDance modifications.
