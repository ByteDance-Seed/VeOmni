# Qwen4-Exp integration status

Supported in this correctness stage:

- GPU and NPU VLM supervised fine-tuning with Ulysses sequence parallelism, plus
  GPU context parallelism (`cp_size > 1`).
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

- Cache prefill/decode, HSDP replicas of persistent PLE, NPU context
  parallelism, and hybrid CP × Ulysses remain unsupported.
- The default `eager` QSA backend expands compact indices to a dense mask and
  materializes the `[B, H_q, S_local, S_global]` attention-score tensor. Use
  `qsa_attention_implementation=tilelang` (NVIDIA CUDA, bf16) for the 16K
  performance gate; the TileLang backend covers training forward/backward only
  and raises on KV-cache or additive-mask layouts. Nonzero attention dropout
  falls back to eager QSA and therefore loses the sparse kernel's memory
  advantage.
- Ulysses/CP GatedDeltaNet requires non-eager, varlen-capable causal-conv and
  chunk gated-delta-rule kernels. GatedDeltaNet head counts must be divisible by
  the active `ulysses_size`/`cp_size`; QSA attention has no head-divisibility
  constraint because it keeps all query heads local and all-gathers only KV.
- PLE n-gram history and its dilated convolution reset at explicit
  `cu_seq_lens_q` packed-sample boundaries. The halo-only fallback remains for
  direct component calls that do not provide packing metadata.
- Distributed PLE training expects pretrained or DCP weights. Initializing from
  scratch after PLE parameters become DTensors is not supported by the upstream
  Hugging Face initializer.
- The real checkpoint schema and shard reads plus two-process lookup/backward
  semantics are validated without materializing the 95 GiB table. A real
  multi-node load-and-train smoke run remains the final deployment gate.

## Two-device pipeline regression

Run the daily toy regression with:

```bash
pytest -s tests/e2e/test_e2e_parallel.py::test_qwen4_exp_training_smoke
```

On a host with at least two CUDA or NPU devices, the test uses VLM dummy data,
`ple_size=2`, `ep_size=2`, and `ulysses_size=2` to cover FSDP2 streaming load
and a fused-MoE forward/backward training smoke with finite loss and gradient
norm. It does not load the released 95 GiB PLE table.
