# Qwen4-Exp integration status

This directory is an internal-validation integration for the experimental
`qwen4_exp` model from Transformers commit
`93d1bcfbd2af5798e2f66bf7955e31a537902b64`.

Supported in this first stage:

- GPU and NPU VLM supervised fine-tuning with `ulysses_size=1` and `cp_size=1`.
- Eager or SDPA QSA correctness paths.
- VeOmni fused cross-entropy and fused MoE dispatch.
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

- Ulysses/context sequence parallelism is rejected because PLE n-gram context
  and QSA global token indices need dedicated distributed semantics.
- The production QSA kernel is not integrated. Upstream eager/SDPA QSA builds
  dense masks and is suitable only for short correctness validation.
- Distributed PLE training expects pretrained or DCP weights. Initializing from
  scratch after PLE parameters become DTensors is not supported by the upstream
  Hugging Face initializer.
- The real checkpoint schema and shard reads plus two-process lookup/backward
  semantics are validated without materializing the 95 GiB table. A real
  multi-node load-and-train smoke run remains the final deployment gate.

## Two-device pipeline regression

Run the daily toy regression with:

```bash
pytest -s tests/e2e/test_qwen4_exp_pipeline.py
```

On a host with at least two CUDA or NPU devices, the test uses VLM dummy data,
`ple_size=2`, and `ep_size=2` to cover FSDP2 streaming load, fused MoE
forward/backward, PLE and expert AdamW updates, DCP save/resume, finite loss,
and non-zero PLE/expert gradients. It does not load the released 95 GiB PLE
table.
