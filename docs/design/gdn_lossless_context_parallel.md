# Lossless GDN and KCP Context Parallelism

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

The implementation targets packed text training on GPU and Ascend NPU with a
power-of-two CP size. `state_passing_lossless` is supported on both hardware
paths; `kcp` is Ascend-only. Planner/CPU distributed-oracle coverage includes
CP2/4/8/16. Full attention uses causal Ring CP; GDN uses lossless chunk
ownership plus state passing or KCP. The following GDN inputs fail closed:
selector without CP, non-packed batches, attention dropout,
sliding-window/softcap, non-causal or multimodal/cross-attention, `kcp` on GPU,
a missing hardware-specific fused-attention backend, and `cp_size > 1` without
an explicit lossless GDN selector. Generic Ring CP is not yet exposed as a
production configuration; this prevents the collator from silently sharding
tokens for an attention path that never executes Ring communication.

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
- `gdn_cp_runtime_evidence.snapshot()` returns a public typed identity and
  A2A/P2P/AG enter-exit-error counters without log-marker parsing;
- generated Qwen3.5 and Qwen3.5-MoE GPU/NPU modeling files must pass
  `patchgen --check` and must never be edited directly.

The CPU reference attention backend exists only for unit tests. Long-sequence
training requires a hardware-specific fused-attention backend: VeOmni
FlashAttention SP on GPU or fusion attention on Ascend NPU. The Ring scheduling
helpers are adapted from MindSpeed under BSD-3-Clause; source files preserve
Huawei attribution and identify ByteDance modifications.
