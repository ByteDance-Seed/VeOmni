# FP8 MoE training with DeepGEMM on Hopper and Blackwell

`moe_implementation: deepgemm_fp8` selects native FP8 grouped GEMMs for
routed SwiGLU experts' forward, activation gradients (dgrad), and weight
gradients (wgrad). It implements the existing `moe_experts/standard` contract
and is opt-in. It does not require Transformer Engine.

## Configuration

For a model using the standard SwiGLU expert contract:

```yaml
model:
  ops_implementation:
    moe_implementation: deepgemm_fp8
    qat_implementation: none
```

The model's existing `VeomniOp` handle resolves the backend at construction.
Other operators keep their configured implementations. The default MoE backend
is unchanged. Use trainable BF16 expert parameters, or FSDP2 BF16 parameter
mixed precision with FP32 master weights. A packed inference checkpoint must
first pass through its model's existing dequantization/weight-loading path.

Keep model-side QAT disabled for this native FP8 path. The backend receives
already prepared tensors and cannot undo a fake-quantization pass performed
by a model before dispatch. This setting is independent of rollout/export
quantization.

The mainline DeepSeek V4 expert module is covered by a real forward/backward
integration test, using its existing merged weights and routing-score cast.
This is operator integration, not full-model convergence validation.

## Requirements and tensor contract

- NVIDIA Hopper SM90 (H20/H100/H200) or Blackwell SM100 (GB200). Other
  architectures fail explicitly at op resolution. This implementation uses
  128-element/block FP8 quantization; it does not provide MXFP8-32 or FP4.
  Use one CUDA device per process: the pinned DeepGEMM JIT caches device-bound
  kernel handles without a device key, including the SM100 scale-packing kernel.
  The backend rejects a second device explicitly. Forward and backward establish
  the operand device context even when a different device is currently selected.
  Standard one-worker-per-GPU EP/FSDP2 training satisfies this restriction. Other
  DeepGEMM calls in the same process must also use that worker's device.
- BF16 hidden states `[tokens, H]` and expert weights. Gate/up weights can be
  merged `[E, 2*I, H]` or split into two `[E, I, H]` tensors; down weights are
  `[E, H, I]`. Hidden size `H` and intermediate size `I` must be positive
  multiples of 128. `E` is the local expert count when EP is enabled.
- Routing weights and int64 expert ids have shape `[tokens, top_k]` and share
  the hidden states' device. Routing weights retain the dtype supplied by the
  model and are applied after the down projection; output and hidden gradients
  are BF16. Standard model adapters may cast router scores before the op.
- Uneven loads, empty experts, empty batches and repeated expert ids are
  supported. The mainline EP dispatcher counts unique token/expert pairs and
  combines repeated slot weights. `assume_distinct_experts` remains an accepted
  contract hint; this backend does not require it.
- First-order autograd and non-reentrant activation checkpointing are supported.
  Full-graph `torch.compile` and CUDA graph capture are unsupported because
  Hopper's K-grouped launch requires token counts on the host.
- FSDP2 parameter communication and EP token all-to-all retain their configured
  precision. This change does not add FP8 all-gather, quantized optimizer state,
  or a persistent FP8 parameter cache.

## Installation

`uv sync` with the `gpu` extra installs the pinned DeepGEMM wheel from
[verl-wheelhouse](https://github.com/verl-project/verl-wheelhouse). The wheels
package commit `5081f8513af57e61696de4f4f04b0e707ffb503e` as
`deep-gemm==2.8.2+5081f85`, built for PyTorch 2.11.0 and CUDA 13.0.2.
`pyproject.toml` selects the release asset for Python 3.11 or 3.12 and Linux
x86_64 or aarch64; `uv.lock` records its SHA256. CPU-only and NPU environments
do not install this package.

```bash
uv sync --extra gpu --dev --locked
```

These are native Linux wheels built on Ubuntu 24.04, not portable manylinux
wheels. Their host extension requires glibc 2.38 or newer and libstdc++ providing
`GLIBCXX_3.4.31` on x86_64 or `GLIBCXX_3.4.29` on aarch64. The supported
NGC 25.11 / Ubuntu 24.04 environment satisfies these requirements.

Installation uses the prebuilt host extension, but the first invocation of a
new GEMM specialization still JIT-compiles a GPU kernel. Keep the CUDA 13
NVCC toolkit and a C++20 compiler/standard library available (validated with
GCC 13), and preserve the host's CUDA compatibility library paths. The wheel
includes the DeepJIT/CUTLASS headers; a DeepGEMM source checkout is not required.
No Transformer Engine dependency is added.

The backend remains opt-in. Missing dependencies and unsupported devices raise
instead of falling back to BF16. Merely importing `veomni.ops` does not import
DeepGEMM or its Triton kernels.

## Computation and memory

All GEMM operands use E4M3 FP8. Hopper uses FP32 scales
`max(amax, 1e-4) / 448`, without power-of-two rounding. Blackwell rounds these
scales upward to powers of two **before** quantizing the payload, then lets
DeepGEMM pack them as UE8M0. Rounding only scales at GEMM launch would silently
change the represented values. The numerical recipes consequently differ
between architectures; both are compared with their corresponding upstream
quantizers and a common BF16 reference. For `Y = X @ W.T`,
where `X` has shape `[T, H]`:

| Operation | Activation/gradient quantization | Weight quantization |
| --- | --- | --- |
| Forward | `X`: 1×128 along hidden channels | 128×128 blocks |
| Dgrad | `dY`: 1×128 along output channels | Transposed quantized square blocks |
| Wgrad | `X` and `dY`: independent 128×1 groups along tokens | Not used |

Wgrad operands are quantized from BF16 source tensors, not from a transpose of
forward FP8 values. The input's token-axis representation is created during
forward and saved for wgrad when it is smaller than the BF16 source, allowing
that source activation to be released when its other consumers finish. Sparse
groups retain BF16 when token padding and FP32 scales would make FP8 larger,
and quantize it during backward instead. Output gradients are quantized during backward.
Each expert is padded independently to DeepGEMM's group alignment (128 tokens
by default, a multiple of 128); scale
groups never cross expert boundaries. Triton constructs packing metadata and
quantized operands. DeepGEMM M-grouped GEMMs produce BF16 forward/dgrad results;
its K-grouped GEMM accumulates wgrad in FP32. SM90 materializes an external FP32
workspace before conversion; SM100 casts directly to BF16 in the GEMM epilogue
with `c=None`. Gradients use the straight-through quantization approximation.

The SM90 wgrad FP32 workspace has a 256 MiB target and is reused across batches of
complete experts. A single expert larger than the target uses one expert's
workspace. Each expert retains its full FP32 token reduction. On SM90, column operands concatenate transposed expert segments for the NT
wgrad API. On SM100 they retain contiguous token-major `[sum_T,H]` storage
and `[sum_T/128,H]` scales for the TN wgrad API. SM100 launches up to 128
experts per call to satisfy the upstream scale-packing limit, without an
external FP32 workspace, its initialization, or a separate output cast.
Column operands are quantized once and sliced as views. Empty groups remain
zero, including entirely empty chunks and the final partial chunk.

Fused Triton SwiGLU forward/backward kernels save only the original BF16 gate/up
projection and recompute pointwise values, preserving eager BF16 rounding and
the inclusive clamp-boundary derivative. Forward-only FP8 weights and padded
dgrad temporaries are released after their last use. For contiguous BF16
`Parameter` inputs, forward saves a reference to the existing parameter and
produces its FP8 transpose only during dgrad. FSDP2 still releases the parameter
storage on reshard and repopulates it before backward. Other weight tensors
(including casts, QAT outputs, split-weight concatenations and layout copies)
retain an invocation-local FP8 transpose, avoiding retention of a larger BF16
temporary. Only the required orientation is quantized at each stage. Weight
transposes are released after dgrad, before allocating wgrad operands and workspace, unless
the caller requests `retain_graph=True`. EP dispatch masks and local input
references are released before combine. Saved weight transposes
belong to one invocation, so optimizer steps and FSDP resharding cannot leave a
stale module-level quantized cache. FP8 GEMMs alone do not guarantee a lower
whole-model memory peak.

Routed token counts remain runtime arguments to the row-quantization and
SwiGLU kernels, so changing these counts does not create a new Triton binary
for each value. Channel dimensions and tensor layouts still select specialized
kernels.

## Tests

Run on Hopper or Blackwell SM100 after syncing the GPU environment:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/pytest -q \
  tests/ops/moe_experts/ tests/ops/base/

OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/pytest -q \
  tests/models/transformers/deepseek_v4/test_deepseek_v4.py \
  -k deepgemm
```

Coverage includes quantizer parity with the pinned DeepGEMM helpers,
forward/dgrad/wgrad numerical checks, SwiGLU/clamp gradients, empty and repeated
routes, activation checkpointing, EP/FSDP2, and optimizer updates. Registry and
lazy-import checks remain available without the GPU dependency.

## Implementation entry points

- [Registry](../../veomni/ops/kernels/moe_experts/__init__.py)
- [Standard expert wrapper](../../veomni/ops/kernels/moe_experts/standard/deepgemm.py)
- [Quantizers and training autograd](../../veomni/ops/kernels/moe_experts/shared/deepgemm_fp8.py)
- [Numerical tests](../../tests/ops/moe_experts/test_deepgemm_fp8.py)
- [Distributed tests](../../tests/ops/moe_experts/test_deepgemm_fp8_distributed.py)
