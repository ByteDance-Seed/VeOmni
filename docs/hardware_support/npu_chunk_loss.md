# NPU fast path for standard chunk cross entropy

The registered `cross_entropy_loss / standard / chunk_loss` implementation
uses native Ascend cross entropy for supported NPU inputs. No model patch or
new implementation name is needed. Label shifting and sequence-parallel
reduction remain in the model loss adapter.

The fast path removes `labels == ignore_index` rows before the vocabulary
projection, then computes FP32 summed CE from FP16, BF16 or FP32 projection
inputs. A first-order autograd adapter supports `torch.func.grad_and_value`.
An all-ignored chunk returns a graph-connected zero. The shared denominator
accepts an integer or Tensor; an explicit zero uses one, matching eager CE.
Ignored rows are removed before calling the kernel with a fixed negative
ignore sentinel.
Custom ignore indices validate the remaining target range before native CE,
so the fixed sentinel cannot mask illegal targets. This validation adds a
device synchronization for non-default ignore indices.

Full or factored logits bias, per-token observation, compilation, non-int64
labels, mismatched devices/dtypes and more than 200,000 flattened rows per
chunk use the existing eager implementation. The row bound is conservative:
it applies before compaction. Native errors are not caught or silently retried.

The existing `SavedState`, selective-gradient handling and activation-offload
hook suspension remain in the outer chunk implementation. This adapter
supports first-order reverse differentiation; it does not add higher-order
gradients, forward-mode AD or vmap support.

## Verification

Run the native parity matrix and existing offload regressions on the project's
supported NPU stack:

```bash
python -m pytest tests/ops/loss/test_npu_chunk_loss.py \
  tests/ops/loss/test_chunk_loss_offload.py
make quality
```

The three-way synchronous microbenchmark separates valid-row compaction from
native CE:

```bash
python scripts/profile/benchmark_npu_chunk_ce.py --output /tmp/npu_chunk_ce.json
```

It reports median chunk forward/gradient time, additional peak allocated
memory and numerical differences against full eager projection. It does not
measure full training, FSDP2 communication, SP collectives or activation
offload, and results must include the accelerator and software versions.
