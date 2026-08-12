# Lossless GDN Context Parallelism

This document defines the production correctness foundation for Qwen3.5
GatedDeltaNet context parallelism. It deliberately does not include the
experimental KCP affine optimization.

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

The implementation targets packed text training on Ascend NPU with a
power-of-two CP size. Planner/CPU distributed-oracle coverage includes
CP2/4/8/16; an Ascend/HCCL full-gradient ladder and 512K capacity/performance
run remain release gates before those sizes can be described as production
validated. Full attention uses causal Ring CP; GDN uses lossless chunk
ownership plus state passing. The following inputs fail closed: CP without the
selector, selector without CP, non-packed batches, attention dropout,
sliding-window/softcap, non-causal or multimodal/cross-attention, CUDA Ring CP,
and missing NPU fusion attention.

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
- generated Qwen3.5 and Qwen3.5-MoE GPU/NPU modeling files must pass
  `patchgen --check` and must never be edited directly.

The CPU reference attention backend exists only for unit tests. Long-sequence
training requires the Ascend fusion-attention backend. The Ring scheduling
helpers are adapted from MindSpeed under BSD-3-Clause; source files preserve
Huawei attribution and identify ByteDance modifications.
