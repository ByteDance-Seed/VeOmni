# Qwen4-Exp integration status

Supported in this correctness stage:

- GPU and NPU VLM supervised fine-tuning with Ulysses sequence parallelism and
  context parallelism (`cp_size > 1`). NPU context parallelism is enabled only
  for Qwen4-Exp and requires NPU-compatible attention and GatedDeltaNet backends.
  Combined Ulysses/CP uses the explicit opt-in described below.
- GatedDeltaNet sequence-to-head all-to-all using the existing Qwen3.5
  causal-conv and gated-delta kernels over the active SP/CP group.
- VisionModel follows Qwen3.5's SP path: each rank runs patch embedding and
  merger on its local visual-token shard, vision attention scopes `use_parallel_state`
  to a temporary Ulysses SP state so standard attention performs sequence/head
  exchange over the active SP/CP group, and only the merged features are
  gathered for the global placeholder scatter.
- CP-style QSA selection plus main-attention KV all-gather. The indexer uses
  local block pooling, compressed-row all-gather, and local queries to return
  local `[B, S_local, K]` compact global token indices; `Qwen4ExpTextAttention`
  keeps `query_states` local (`[B, H_q, S_local, D]`) with all query heads intact
  and all-gathers only `key_states` and `value_states` (`[B, H_kv, S_global, D]`).
  `qsa_attention_implementation=eager` expands compact indices to a local-query
  `[B, 1, S_local, S_global]` mask, while `qsa_attention_implementation=tilelang`
  runs the gather-based TileLang sparse-attention kernel
  (`veomni/ops/kernels/qsa_attention/`, bf16, NVIDIA CUDA) directly on the local
  query shard and all-gathered KV with full GQA head grouping preserved.
  Explicit cu-seqlens isolate packed samples consistently for SP/CP=1 and SP/CP>1.
- PLE n-gram token halos and differentiable dilated-convolution halos across
  Ulysses or context-parallel shard boundaries, while persistent PLE table
  routing remains on the independent `ple_fsdp × ple` mesh. Packed metadata is
  computed once per model forward; n-gram histories reset at local sample
  offsets, and convolution correction is bounded to the first
  `(kernel_size - 1) * dilation` outputs of each sample in fixed-size chunks.
  Interior tokens retain the original convolution and SP halo, with no added
  collective or checkpoint parameter.
- VeOmni fused cross-entropy and fused MoE dispatch, with sequence-local router
  logits and padding masks for auxiliary load-balancing loss.
- Concurrent PLE and MoE expert parallelism: PLE tables use the persistent
  two-dimensional `ple_fsdp × ple` layout while expert tensors use the
  independent `ep_fsdp × ep` layout.
- Scalable loading and training of the released ~95 GiB PLE table:
  - preserve its 128 checkpoint-native embedding shards instead of
    concatenating them;
  - persistently shard every table by rows over `ple` and columns over the
    complementary `ple_fsdp` dimension;
  - stream only the local row-by-column rectangle and zero-pad final rows;
  - keep PLE weights outside FSDP2, eliminating their forward/backward
    parameter all-gathers;
  - route lookup requests and differentiable column slices over the flattened
    PLE mesh, so every rank may process a different sample.
- Explicitly discarding `mtp.*` checkpoint tensors; MTP loss is not supported.

The internal example uses `ple_size=8`, `ep_size=4`,
`broadcast_model_weights_from_rank0=false`, and
`ep_sharded_stream_load=true`. The world size and every padded PLE shard row
count must be divisible by `ple_size`; the number of experts must be divisible
by `ep_size`. PLE and EP may be enabled together because their parameter sets
and communication meshes are independent.

The implementation details and constraints of this Qwen4-Exp-only local
parallel path are documented in
[Qwen4-Exp PLE Two-Dimensional Parallelism](../../../../docs/design/qwen4_exp_ple_2d_parallelism.md).

Known limitations:

- Cache prefill/decode and HSDP replicas of persistent PLE remain unsupported.
  Combined CP × Ulysses requires the explicit Qwen4-Exp opt-in described below.
- The default `eager` QSA backend expands compact indices to a dense mask and
  materializes the `[B, H_q, S_local, S_global]` attention-score tensor. Use
  `qsa_attention_implementation=tilelang` (NVIDIA CUDA, bf16) for the 16K
  performance gate; the TileLang backend covers training forward/backward only
  and raises on KV-cache or additive-mask layouts. Nonzero attention dropout
  falls back to eager QSA and therefore loses the sparse kernel's memory
  advantage.
- Ulysses/CP GatedDeltaNet requires non-eager, varlen-capable causal-conv and
  chunk gated-delta-rule kernels. Both GatedDeltaNet key and value head counts
  must be divisible by `ulysses_size` for Ulysses-only execution, by `cp_size`
  for CP-only execution, and by `ulysses_size * cp_size` for combined execution.
  QSA attention has no head-divisibility
  constraint because it keeps all query heads local and all-gathers only KV.
  When `chunk_gated_delta_rule_implementation=flash_qla`, GatedDeltaNet
  preserves grouped Q/K heads (`num_k_heads < num_v_heads`) directly into the
  FlashQLA chunk kernel instead of expanding Q/K with `repeat_interleave`.
- PLE n-gram history and its dilated convolution reset at explicit
  `cu_seq_lens_q` packed-sample boundaries. The halo-only fallback remains for
  direct component calls that do not provide packing metadata.
- Distributed PLE training expects pretrained or DCP weights. Initializing from
  scratch after PLE parameters become DTensors is not supported by the upstream
  Hugging Face initializer.
- The real checkpoint schema and shard reads plus two-process lookup/backward
  semantics are validated without materializing the 95 GiB table. A real
  multi-node load-and-train smoke run remains the final deployment gate.

## Compact QSA indexing on Ascend

The indexer uses INT8 for NPU visibility gathers (the resulting mask is still
boolean) and bounded INT32 position arithmetic. Top-k scores and floating
attention arithmetic are unchanged. Positions including compression padding
must fit INT32, matching the output-index contract.

A two-entry host-only LRU caches immutable block metadata keyed by the complete
packed boundaries, compression ratio, local shard length and SP rank/group size.
Blocks are bucketed by owner in one pass, and counts stay on the host. One model
forward prepares the device metadata before the decoder loop and reuses it across
QSA layers and checkpoint recomputation. Only rank-local query segment maps are
materialized. Device tensors are not cached across forwards; direct indexer
calls prepare their own metadata through the same validated path.

## Two-device pipeline regression

Run the daily toy regression with:

```bash
pytest -s tests/e2e/test_e2e_parallel.py::test_qwen4_exp_training_smoke
```

On a host with at least two CUDA or NPU devices, the test uses VLM dummy data,
`ple_size=2`, `ep_size=2`, and `ulysses_size=2` to cover FSDP2 streaming load
and a fused-MoE forward/backward training smoke with finite loss and gradient
norm. It does not load the released 95 GiB PLE table.

### Combined Ulysses/CP (opt-in)

Set `model.accelerator.allow_hybrid_cp=true` to opt into a combined topology,
for example `ulysses_size=8, cp_size=2` on 32 ranks with DP=2. Other model
families retain the combined-topology rejection. The collator's contiguous
rank order is the flattened CP×Ulysses group order; QSA keeps local queries
and gathers KV over that complete group. GDN exchanges sequence for heads
over the same group. The combined degree must divide both GDN key and value
head counts. This is not recurrent-state context sharding for GDN.

Vision uses FlashAttention/Ulysses over that complete sequence group, with
local patch rows and global varlen metadata. Eager/SDPA vision remains
unsupported under SP. Combined degree 32 is not supported by a real model
with only 16 GDN key heads; enabling the flag does not bypass head constraints.
Hardware numerical and cold-resume qualification are recorded per profile,
not implied by configuration parsing.
