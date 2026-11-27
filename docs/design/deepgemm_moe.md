# FP8 training with DeepGEMM on Hopper and Blackwell

`deepgemm` is one implementation name that three GEMM families each register on
their own field:

| Field | What it covers | Contract |
| --- | --- | --- |
| `moe_implementation` | routed SwiGLU experts: forward, dgrad, wgrad | `moe_experts/standard` |
| `linear_implementation` | dense (`standard`) and block-diagonal (`grouped`) projections | `linear/standard`, `linear/grouped` |
| `gated_mlp_implementation` | a dense gated MLP's gate/up/down projections | `gated_mlp/standard` |

All three run native FP8 GEMMs on the tensor cores and implement the existing
contracts, so they are opt-in and a model's `VeomniOp` handles resolve them
without a modeling change. They do not require Transformer Engine.

Where `deepgemm` sits among the low-precision rows:

| Row | Arithmetic |
| --- | --- |
| `qat` | fake-quantizes the operands, multiplies in the model dtype — emulation |
| `torchao` `dtype=fp8` | native `aten::_scaled_mm` by default (`emulated=False`); `emulated=True` falls back to fake quantization for hosts below SM89 |
| `torchao` `dtype=nvfp4` / `mxfp4` | fake quantization (`emulated=True`; native FP4 is not implemented) |
| `deepgemm` | DeepGEMM's blockwise FP8 **grouped** GEMM, covering forward, dgrad and wgrad |

So `torchao`'s FP8 mode is *not* emulation, and calling it one would misrepresent a
genuine native path. What distinguishes `deepgemm` is the recipe and the shape it
covers: 128-wide blockwise tiles driven through DeepGEMM's grouped-GEMM API, with
dgrad and wgrad on FP8 as well, and one row that serves routed experts, dense
projections and dense gated MLPs alike.

On each field these are different values, so a single GEMM is never rounded twice;
across fields nothing links them, and a mixed selection is a valid recipe.

## One restriction: the gated MLP is SiLU-family only

`gated_mlp_implementation: deepgemm` cannot serve a model whose `hidden_act` is
outside `{silu, swish}`. Its Triton kernel fuses `aten.silu` and
`aten.silu_backward` into the grouped GEMM's forward and backward, and `swish` is the
same function under its other name, so both are served. Anything else — `gelu`,
`gelu_new`, `relu` — has no fused row to run. `render` refuses the selection and
names `eager` as the way to keep the fallback, so such a model selects
`gated_mlp_implementation: eager` and trains that family in the model dtype. Teaching
the kernel another activation is tracked in #1430.

The routed experts have the same restriction for the same reason. `moe_experts`
rows take `activation=` too, and `deepgemm` refuses anything `is_silu` rejects
(`silu` / `swish`), because its grouped GEMM also fuses the activation — the error
names the activation and says the FP8 GEMM would have to be dropped to substitute
another.

`linear` is the one row among the three with no such limit: it takes no activation.

## Configuration

For a model using the standard gated-expert contract:

```yaml
model:
  ops_implementation:
    moe_implementation: deepgemm
```

Every GEMM family on real FP8 tensor cores. For a checkpoint whose config sets
`expert_dtype: "fp4"` (such as DeepSeek-V4 Flash), name `dtype: fp8` explicitly on the
MoE field so the row accepts the call instead of rejecting `dtype="fp4"` — the rows
validate the format rather than ignoring it, and DeepGEMM ships FP8 kernels only:

```yaml
model:
  ops_implementation:
    linear_implementation: deepgemm
    gated_mlp_implementation: deepgemm
    moe_implementation: deepgemm
    # Stored KV / indexer rounding is a separate decision.
    kv_round_implementation: triton
```

On an `expert_dtype: "fp4"` checkpoint the MoE field carries the format explicitly,
because that config would otherwise resolve `dtype="fp4"` and be refused:

```yaml
model:
  ops_implementation:
    moe_implementation:
      name: deepgemm
      kwargs: {dtype: fp8}
```

The model's existing `VeomniOp` handle resolves the backend at construction.
Other operators keep their configured implementations. The default MoE backend
is unchanged. Use trainable BF16 expert parameters, or FSDP2 BF16 parameter
mixed precision with FP32 master weights. A packed inference checkpoint must
first pass through its model's existing dequantization/weight-loading path.

`deepgemm` and `qat` are two values of the same `*_implementation` field, so one
GEMM cannot be quantized twice. The other families are independent: selecting
`qat` for `linear_implementation` or `gated_mlp_implementation` alongside
`deepgemm` experts is valid, because the routed-expert kernel takes the experts'
merged slab and the dense families are separate GEMMs with their own fields.
Combine them freely; this is independent of rollout/export quantization.

The dense rows take the argument contract of their families rather than a
call-shape of their own: `linear` rows take the projection **module** (so
`resolve_projection` can refuse a wrapped projection or an `nn.Linear` subclass
such as the `FSDPLinear` that `basic_modules` produces), and `gated_mlp` rows take
the MLP **block** plus `activation=`. `deepgemm` validates `dtype` and
`block_size` instead of ignoring them, because it has exactly one recipe (FP8,
128-wide tiles); accepting another value would let one field mean two formats
depending on which row the config selected.

LoRA is unsupported on every `deepgemm` row, for one reason: it quantizes the
weight it multiplies, so it reads `module.weight` instead of calling the module, and
a wrapped projection keeps the real weight at `base_layer`. The expert path says so
through the LoRA wrappers, which raise before moving the base parameters; the dense
families say so through `resolve_projection` / `render`, which refuse a wrapped
projection. Either way it is a refusal rather than a silent switch to BF16 eager
execution. For MoE-LoRA select a supported backend such as `fused_triton` on GPU;
for a dense adapter set `linear_implementation` / `gated_mlp_implementation` to
`eager`.

The mainline DeepSeek V4 expert module is covered by a real forward/backward
integration test, using its existing merged weights and FP32 router scores.

The dense `linear` rows are covered in two places, and the two compare against
different things: **forward** against an independently re-quantized reference (the
values a deployment FP8 GEMM would produce), and **dgrad / wgrad** against a BF16
reference GEMM (the operands are quantized, so a straight FP8 reference would only
reproduce the same rounding). Row counts include one divisible and one not divisible
by the 128-column alignment.

The `gated_mlp` row splits the same way for a different reason: its clamp means the
plain BF16 reference does not reproduce the forward, so forward is checked against
the quantized reference and dgrad / wgrad against a BF16 autocast reference.

All of these tests need SM90/SM100 hardware and `deep_gemm`, so they skip on the
CPU-only hosts that run part of CI — a green CI run does not by itself prove the
rows execute. After changing `GroupedLinear`, the layout, or a quantizer, run
`tests/ops/moe_experts/test_deepgemm.py` (expert and dense rows) and
`tests/ops/linear/test_linear.py` on a Hopper or Blackwell host.

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
  are BF16.
- Uneven loads, empty experts, empty batches and repeated expert ids are
  supported. The mainline EP dispatcher counts unique token/expert pairs and
  combines repeated slot weights. `assume_distinct_experts` remains an accepted
  contract hint; this backend does not require it.
- First-order autograd and non-reentrant activation checkpointing are supported.
  Full-graph `torch.compile` and CUDA graph capture are unsupported because
  Hopper's K-grouped launch requires token counts on the host.
- Parameter all-gather uses its configured precision by default. The optional
  FP8 transport below applies only to routed expert weights. EP token all-to-all,
  gradients and optimizer state keep their configured precision.

## Optional FP8 parameter all-gather

```yaml
model:
  ops_implementation:
    moe_implementation: deepgemm  # Use {name: deepgemm, kwargs: {dtype: fp8}} on expert_dtype="fp4" checkpoints (e.g. DeepSeek-V4 Flash)
  accelerator:
    fsdp_config:
      expert_weight_fp8_all_gather: true
      expert_fsdp_shard_dim: 0
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
```

This opt-in path uses native Torch 2.11 FSDP2 extensions. It quantizes each
local expert weight shard before communication, gathers E4M3 payloads and
FP32 block scales, and passes them directly to DeepGEMM. Payload copies and
transposes use byte views to avoid unnecessary float8 conversion kernels;
the bits and quantization recipe are unchanged. Dense layers,
routers, shared experts, EP token communication and gradient reduction are
unaffected. Master parameter shards and optimizer/checkpoint state remain in
their original precision. The quantizer rounds FP32 masters to BF16 inside
the kernel, matching the default BF16 parameter all-gather's numerical recipe.

For each 128x128 block, transport contains 16384 FP8 bytes plus a four-byte
scale, compared with 32768 BF16 bytes: about half the parameter payload.
Scales use separate small collectives because Torch 2.11's nonzero-dimension
shard copy-out assumes every extension output has the parameter's shape.
Extra collective launches and local quantization affect latency; this byte
reduction is not a promise of a 2x training speedup.

Forward consumes the gathered FP8 weight directly. Dgrad transposes its
payload and its scale grid: square 128x128 blocks preserve their exact values
under transpose. FSDP frees and repopulates payload/scale storage on reshard
and unshard, including backward and activation recomputation. Local shards
are requantized on every unshard, so optimizer updates cannot leave an
across-step FP8 cache stale.

Synchronous activation offload remains supported: it skips the gathered FP8
weight container and offloads ordinary saved activations as usual. FSDP owns
the weight payload/scale storage and restores it before backward; activation
offload must not copy that container to CPU or count it against its GPU budget.

`expert_weight_fp8_all_gather: true` applies only to modules whose actual `veomni_moe.impl`
is `deepgemm`. If none match, the flag is silently ignored, for standalone
models, composed models and individual children alike. FP8-specific policy
checks run only after this selection, so a dense child can inherit the flag
while using DDP, FP16, CPU offload or compilation under its usual backend
rules. The flag itself must always be a boolean.

This option controls routed-expert **parameter** all-gather during FSDP2
unshard, including forward/backward prefetch and backward re-gather after
resharding. EP token all-to-all, gradient reduce-scatter, and other parameters'
all-gather retain their configured behavior. Expert FSDP shard size one has
no cross-rank expert parameter all-gather to compress.

Constraints for selected experts:

- Two merged expert weights `[E,2*I,H]` and `[E,H,I]`, selected by the actual
  module's `veomni_moe.impl == "deepgemm"` handle.
- BF16 FSDP mixed precision, CUDA, contiguous BF16/FP32 master shards.
- Even sharding, with complete 128x128 blocks inside every shard. The default
  EP FSDP `Shard(1)` layout must divide both `2*I` and `H` into multiples of
  128. Unsupported shapes fail before the first training parameter all-gather.
- No FSDP parameter CPU offload, integer partial resharding, `torch.compile`, or CUDA graphs.
  Register the hooks after wrapping and initial weight materialization, before
  the first forward. In-place checkpoint loading is supported; replacing
  parameters or moving the wrapped model after registration is unsupported.

On eight GPUs, EP8 has expert FSDP size one and no expert all-gather to reduce.
Use EP4 with FSDP size two to measure this path locally, or sixteen GPUs for
EP8 with FSDP size two. Compare identical parallel layouts and include the
all-gather and gradient reduction in the measured forward/backward step.

### Whole-expert FSDP sharding and EP combine

`model.accelerator.fsdp_config.expert_fsdp_shard_dim` optionally overrides
EP expert FSDP placement. `null` (the default) preserves the existing hidden
split/Muon selection, `1` splits expert matrix rows, and `0` shards whole
experts. An explicit value overrides `muon_expert_zero_comm` for EP. The setting
applies to declared EP groups in FSDP2; other parameter groups are unchanged.
Dim 0 requires a dim-0 EP plan with at least one 3D expert stack, and each
matched parameter's local first dimension must divide the expert FSDP size.
Unsupported dim-0 layouts fail during wrapping.

This option works with BF16 GEMMs and DeepGEMM FP8. Whole-expert shards avoid
FSDP's hidden-dimension all-gather copy-out reorder and backward gradient
concatenation. Same-dimension EP/FSDP splits use strided placement metadata to
preserve EP-first expert order. DCP planner metadata records each expert
tensor's source EP/FSDP shard dimensions, group sizes, and global expert order
under `__veomni_expert_layouts__` (format version 1). Entries use flattened
state-dict keys, covering both model weights and optimizer tensors without
adding model state keys or tensor collectives. Canonical checkpoints can
reshard between dim 0 and dim 1; the load target need not match the saved layout.

Older Muon dim-0 checkpoints used two ordinary shard placements and encoded a
permuted global expert order. The load planner now checks the **source**
ordering, including when loading into dim 1 or plain tensors. A source marked
as noncanonical is rejected before tensor reads. For unmarked legacy EP
checkpoints, saved chunks that split a matrix dimension identify safe
different-axis sharding and can load into either target layout. Chunks spanning
all experts are also unambiguous. Expert-only chunks cannot distinguish a
valid EP-only save from the faulty same-axis layout: EP resumes reject these
for both target dimensions and require migration using the original training
layout, for model and optimizer together. Changing the target dimension or
merely adding a canonical tag cannot repair a permuted checkpoint.

Earlier checkpoints carrying the `__veomni_ep_first_layout__=1` marker remain
readable. Low-level DCP callers must use the VeOmni planners to run these
checks. Unmarked legacy exports to plain tensors still require external source
layout information: neither tensor names nor shapes alone identify EP weights.

MoE-LoRA adapter export with a supported BF16 backend also preserves the
actual EP and FSDP shard dimensions when consolidating the adapter into
safetensors. Whole-expert shards therefore export the same global adapter
shapes and expert order as hidden-dimension shards.
Master weights, optimizer state, and gradient reduction dtypes are unchanged.
FP32 reduce-scatter buffers can still dominate
peak memory, so FP8 parameter transport alone need not lower the step peak.

The common EP combine path selects `moe_unpermute/standard/triton` on NVIDIA
GPUs and `eager` on other devices or when deterministic algorithms are enabled. It fuses route weighting into FP32 scatter
accumulation and gathers gradients directly, avoiding expanded weighted-token
and FP32 token copies in forward and backward. Both BF16 and FP8 expert
backends use it. Multiplication retains the promoted input/weight dtype's
rounding before FP32 accumulation; the EP dispatcher restores the original
token dtype after the combine. The router
weight gradient similarly rounds each product before its channel reduction.
Floating-point atomic summation order may differ from the eager scatter.
The operator supports first-order autograd and requires valid token indices.

The implementation is in
[fp8_all_gather.py](../../veomni/distributed/fsdp2/fp8_all_gather.py), with a
[weight container](../../veomni/ops/kernels/moe_experts/shared/fp8_weight.py)
consumed by the existing grouped-linear autograd implementation.

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
- [Quantizers and training autograd](../../veomni/ops/kernels/moe_experts/shared/deepgemm.py)
- [Numerical tests](../../tests/ops/moe_experts/test_deepgemm.py)
- [Distributed tests](../../tests/ops/moe_experts/test_deepgemm_distributed.py)
