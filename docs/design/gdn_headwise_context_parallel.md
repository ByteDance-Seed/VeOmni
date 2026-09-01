# Headwise-Lossless GDN Context Parallelism

This document defines the production correctness contract for Qwen3.5
GatedDeltaNet context parallelism. The supported GDN-specific algorithm is
`headwise_lossless`.

## Configuration and scope

```yaml
train:
  dyn_bsz: true
  accelerator:
    cp_size: 4
    ulysses_size: 2
model:
  ops_implementation:
    gdn_context_parallel_implementation: headwise_lossless
```

Full-attention layers retain their physical Ring/Hybrid layout. Each GDN layer
packs q/k/v/b/a into one equal-split `all_to_all_single`, gathers the canonical
packed sequence, and shards GDN heads over the flattened
CP-major/Ulysses-inner group. One inverse exchange restores the physical token
layout. The ordinary fused GDN forward and backward execute once, with no
recurrent-state approximation or duplicated recurrence.

The production path targets packed causal-text training on Ascend NPU with a
power-of-two CP size. CPU execution is reserved for correctness oracles. CUDA
GDN CP, multimodal or cross-attention CP, attention dropout, sliding-window or
softcap attention, and non-packed batches fail closed.

## Layout contract

1. Each packed sample is padded independently to `2 * CP * Ulysses`.
2. Full attention uses the physical mirrored Ring/Hybrid layout.
3. The flattened sequence-parallel identity is
   `sp_rank = cp_rank * ulysses_size + ulysses_rank`.
4. q/k/v/b/a use one packed sequence-to-head A2A. Head counts may differ for
   GQA, but every head count must be divisible by the flattened SP size.
5. Padding is removed per sample before GDN and restored as zero afterward.
   Repeated CU boundaries remain valid empty samples; no dummy token is added.
6. The inverse output A2A restores the original physical token partition.
7. Rank, topology, metadata, shape, dtype, or device disagreement fails on all
   ranks before the payload collective.

## Correctness contract

- packed/GQA forward and full VJP match a monolithic GDN oracle;
- empty and all-empty samples preserve collective symmetry and zero gradients;
- Ulysses x CP uses CP-major/Ulysses-inner flattened rank order;
- non-reentrant activation checkpointing matches eager forward and gradients;
- only equal-split `all_to_all_single` is used; list `all_to_all` is forbidden;
- Mojo keeps its registered external Q/K normalization and custom VJP;
- AscendC keeps the canonical device/host CU and chunk metadata ABI;
- generated Qwen3.5 dense and MoE files are produced by `patchgen` and must
  pass `patchgen --check`.

## Removed selectors

`state_passing_lossless` and `kcp` were experimental implementations and are no
longer accepted. Configurations using either value fail closed during argument
validation. Use `headwise_lossless` for Qwen3.5 GDN CP. The `disabled` value
continues to mean “no GDN-specific CP”; model-native CP implementations retain
their own layout and metadata contract.
