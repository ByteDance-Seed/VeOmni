# FSDP2 `torch.compile` GPU Test Report

This report summarizes the GPU validation for the FSDP2 per-block
`torch.compile` follow-up to PR #810. Raw Chrome trace files and local driver
scripts were used during validation but are intentionally not committed here.

## Scope

The tested change adds:

- `train.torch_compile.{enable,backend,mode,fullgraph,dynamic}` (defaults:
  `backend="inductor"`, `mode="reduce-overhead"`, `fullgraph=True`; `mode` must
  be `None` when `backend="cudagraphs"`).
- FSDP2 decoder-block forward compilation before `fully_shard()`.
- A per-step `torch.compiler.cudagraph_mark_step_begin()` call when CUDA graph
  compile support is active.
- Guards that currently restrict the feature to CUDA + FSDP2 text training with
  `train.dyn_bsz=True` and `train.pad_to_length=True`.

## Environment

Validation was run on NVIDIA H100 GPUs with CUDA-enabled PyTorch. The end-to-end
training validation used a 2-GPU FSDP2 setup with a small Qwen3 toy model and a
synthetic text dataset to avoid external model or data downloads.

## Test Matrix

| Area | Result |
| --- | --- |
| Unit tests for `veomni.distributed.torch_compile` and argument guards | Passed |
| Argument guard: compile enabled without `pad_to_length` | Raised the expected `ValueError` |
| Decoder-only compilation | Compiled only `*DecoderLayer` blocks; skipped ViT / LM head |
| 2-GPU FSDP2 baseline training | Completed; loss decreased normally |
| 2-GPU FSDP2 training with compile enabled | Completed; compiled the toy decoder layers |
| CUDA graph profiling signal | `cudaGraphLaunch` appeared only in the compile run |

## Functional Results

### Unit tests

```text
14 passed
```

The unit tests covered:

- In-place forward compilation without changing module identity.
- Decoder-block-only selection (`*DecoderLayer` classes) — ViT / LM head are skipped.
- Re-entry into `compile_decoder_blocks` on an already-compiled model is a no-op.
- CUDA-only `cudagraph_mark_step_begin()` behavior.
- Argument validation for dynamic batching and static padding.
- Rejection of multimodal-style data argument classes.

### End-to-end FSDP2 training

A 2-layer Qwen3 toy model was trained with 2-GPU FSDP2 in both baseline and
compile-enabled modes. In the compile-enabled run, VeOmni logged that two
decoder-block forwards were compiled.

Observed behavior:

- Baseline and compile-enabled runs followed the same loss curve for the matched
  steps.
- Peak memory was effectively unchanged between baseline and compile-enabled
  runs.
- The first compile-enabled step included expected Dynamo/Inductor compilation
  and CUDA Graph capture overhead; later steps converged toward steady-state
  runtime.

## Profiling Summary

### 1-GPU API micro-benchmark

A direct micro-benchmark of the new compile helper functions showed the expected
CUDA Graph behavior after compile warmup:

| Metric | Compile effect |
| --- | ---: |
| Active-window wall time | about 33% lower |
| GPU kernel launch count | about 50% lower |
| Host-side CUDA launch count | about 67% lower |
| `cudaGraphLaunch` count | appeared in compile run only |

The top GPU kernels were otherwise comparable between baseline and compile
runs, indicating that the main win came from host-side launch reduction and CUDA
Graph replay rather than changing the underlying math kernels.

### 2-GPU FSDP2 profile

The 2-GPU FSDP2 profile confirmed that compile was active in the real training
path:

| Metric | Compile effect |
| --- | ---: |
| GPU kernel launch count | about 36% lower |
| Host-side CUDA launch count | about 47% lower |
| `cudaGraphLaunch` count | appeared in compile run only |

The active profiling window also captured one-time compile/CUDA Graph replay
and FSDP2 communication effects, so the wall-time comparison from that narrow
window should not be treated as steady-state throughput. The important signal is
that the compiled FSDP2 training path produced CUDA Graph launches while keeping
loss and memory behavior aligned with baseline.

## Conclusion

The feature passed unit tests, argument-guard checks, and GPU end-to-end FSDP2
training validation. The compile path successfully compiled decoder-block
forwards and produced CUDA Graph launch activity in profiling, while
maintaining loss and memory parity with the non-compile baseline.
