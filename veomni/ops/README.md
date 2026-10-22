# `veomni.ops`

`veomni.ops` owns VeOmni's tensor-level operation registry and concrete kernel
implementations. Model-specific input normalization, Hugging Face-compatible
signatures, and loss policy belong in `veomni.models`.

Importing `veomni.ops` registers every built-in op family and applies
the process-wide attention integration from `install.py`.

## Layout

```text
veomni/ops/
├── __init__.py          Register built-in families and apply global integrations
├── registry.py          OpEntry, register_op, resolve_op, VeomniOp
├── platform/            GPU platform constraints plus GPU, NPU, and MLU requirements
├── compound.py          Saved-state helpers for ops that call other raw ops
├── config.py            Installed op-selection config read during model construction
├── install.py           Idempotent process-wide attention integration
├── batch_invariant/     Opt-in ATen patch; not a registered op family
└── kernels/             Tensor-level implementations and registrations
```

Python modules maintained by VeOmni document every module, class, and
callable. Directories named `vendor/` mirror external implementations and
retain their upstream source layout and documentation style.

Registered rows use the identity:

```text
(op, variant, implementation, device)
```

Callers select the public `(op, variant, implementation)` triple. The
registry derives `device` from the row's requirement and resolves the current
device first, followed by a device-agnostic row. A row's `requires` metadata
is then checked for discoverable optional module paths without importing vendor
code. This is a static discovery check; actual vendor initialization happens
when the selected implementation runs.

## Built-in families

| Op | Variants | Implementations |
|---|---|---|
| `attention` | `standard` | `eager`, `sdpa`, and FlashAttention/FlexAttention/MagiAttention/SageAttention names |
| `async_ulysses_qkv`, `async_ulysses_o` | `standard`, `dit` | `eager` orchestration |
| `rms_norm` | `standard`, `deepseek_v4`, `unweighted`, `offset` | `eager`, `liger_kernel`, `torch_npu`, and `triton` where supported; all accept optional `group_size` (fused weighted rows delegate grouped calls to eager) |
| `layer_norm` | `standard` | `eager`, `apex` (`fused_layer_norm_cuda`); nested by async Ulysses QKV when `norm_type="layernorm"` |
| `rope` | `full`, `partial`, `interleave`, `mrope`, `deepseek_v4`, `wan` | `eager`, `liger_kernel`, `torch_npu`, and `triton` where supported; `full` also accepts rank-3 vision layout |
| `swiglu_mlp` | `standard`, `geglu` | `eager`, `liger_kernel` |
| `moe_experts` | `standard`, `gpt_oss` | `eager`, `fused_triton`, `fused_quack`, `fused_npu`, `fused_mlu` as supported by the variant |
| `moe_experts_lora` | `independent`, `shared` | `eager`, `fused_triton`, `fused_npu` |
| `cross_entropy_loss` | `standard` | `eager`, `chunk_loss`, `liger_kernel` |
| `load_balancing_loss` | `standard` | `eager`, `triton` |
| `rms_norm_gated`, `causal_conv1d`, `chunk_gated_delta_rule` | `standard` | `eager`, `fla` (CUDA, MLU, NPU), `flash_qla`, `torch_npu`, `triton` (Triton-Ascend), `fla_npu` as supported by the family |
| `dsa_indexer`, `dsa_attention` | `deepseek_v4`, `glm` | `eager`, `tilelang`, `cann`, `cudnn`, or `flashmla_cudnn` by variant |
| `mhc` | `pre`, `post`, `head` | `eager`, `tilelang`, `cann` (`head` / `cann` is the eager math) |
| `qsa_attention` | `standard` | `eager`, `tilelang` (NVIDIA CUDA), `torch_npu` (Ascend NPU) |

DeepSeek-V4 `dsa_attention` takes the full-resolution and compressed KV
separately, plus `position_ids` and the compressor's `CompressedCandidates`,
and derives visibility itself rather than reading a dense mask. Each row
expands that contract its own way: `eager` and `tilelang` build explicit
candidate slots, where repeated valid indices participate once per occurrence
and sentinel entries contribute no attention mass, while `cann` passes mask
modes to the kernel. With `return_lse=True`, `eager` and `tilelang` return a
detached base-2 log-sum-exp tensor.

Qwen4-Exp `qsa_attention` takes the indexer's compact `[B, S, K]` selection of
global K/V token indices, where `-1` marks an unused slot. The selection
already encodes causality and packed-sample boundaries, so the model computes
it and runs any Ulysses exchange before calling the op. `eager` expands the
selection to a dense mask and is the portable reference. `tilelang` gathers the
selected rows without materializing the quadratic score tensor, which is what
makes 16K-token training feasible. It raises on an attention mask and falls
back to `eager` with a one-time warning for nonzero dropout. Its dK/dV backward
uses KV-owned reduction over sorted reverse CSR tiles instead of floating-point
atomic scatter, at the cost of sorting and a partial-gradient workspace.
`torch_npu` prunes each query chunk's `[lo, hi)` KV interval (128-token
aligned), builds a local boolean mask via `int8` scatter, and calls
`torch_npu.npu_fusion_attention` / `npu_fusion_attention_grad` without forward
replay.

The registry is the source of truth for the exact rows available in a given
revision:

```python
from veomni.ops import OP_REGISTRY

OP_REGISTRY.list_registered("rms_norm", "standard")
OP_REGISTRY.list_available("rms_norm", "standard")
OP_REGISTRY.list_entries("rms_norm", "standard")
```

`list_registered` includes every known implementation. `list_available`
filters those rows using the current device, hardware requirement, and
discoverable optional-module requirements.
`list_entries` returns the complete device-specific rows, including their
descriptions, hardware requirements, and package requirements; an
implementation registered for multiple devices therefore appears more than
once.

FlexAttention is registered for NVIDIA GPUs. Its adapter selects FA4 on SM90+
when supported and otherwise uses Triton. MagiAttention requires NVIDIA SM100+
and is optional; install it with:

```bash
uv sync --extra gpu --extra magi --dev
```

## Registering an op

Each row provides either:

- a raw `forward` and `backward` pair, from which the registry generates a
  `torch.autograd.Function` wrapper; or
- one opaque `wrapper`, for eager PyTorch or a library API that already owns
  its autograd behavior.

Do not provide both forms for one row.

```python
from veomni.ops import register_op
from veomni.ops.platform import GpuKernelRequirement, NvidiaGpuPlatform

register_op(
    "example",
    "standard",
    "eager",
    description="PyTorch reference implementation of the example operation",
    wrapper=eager_example,
)

register_op(
    "example",
    "standard",
    "triton",
    forward=triton_forward,
    backward=triton_backward,
    description="Triton implementation of the example operation",
    requirement=GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=80),)),
    requires=("triton",),
)
```

Descriptions identify the implementation source, algorithm, layout, or other
stable semantic differences. Device and compute-capability support belong in
the hardware requirement, while optional import packages belong in `requires`,
so availability metadata cannot drift into the description.

Implementation names identify the implementation source, not the device it
runs on, so no row is named for the `npu` device alone. An Ascend row is named
after what it calls: `torch_npu`, `triton` for vendored Triton-Ascend, `cann`
for `cann_ops_transformer`, or `fla_npu`. PyTorch math that only runs on NPU is
`eager`. MoE keeps the `fused_npu` name to match the other `fused_*`
group-gemm backends. A library that runs on several devices, such as `fla`,
registers one row per device under its library name.

For a raw pair, `forward` returns `(output, SavedState)`, and `backward`
returns one gradient entry for every positional tensor passed to the generated
wrapper. Tensor inputs must therefore be positional; non-tensor attributes
must be keyword arguments.

```python
from veomni.ops.registry import SavedState


def raw_forward(x, *, scale):
    return x * scale, SavedState((), scale)


def raw_backward(grad_output, saved):
    return (grad_output * saved.metadata,)
```

An explicit non-eager selection never silently falls back. Unknown rows raise
`KeyError`; rows for the wrong device, unmet hardware requirements, or missing
optional packages raise `RuntimeError` during resolution.

## Calling an op from modeling

Models construct a local handle once and call it directly:

```python
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl


self.veomni_rms_norm = VeomniOp(
    "rms_norm",
    "standard",
    resolve_op_impl("rms_norm_implementation"),
)

hidden_states = self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)
```

`VeomniOp` resolves its row at construction, is interned by the resolved
triple, and always calls the row's wrapper. `models.build_foundation_model`
installs the `OpsImplementationConfig` object in `ops/config.py` before
constructing the model. Changing the installed config later does not
retarget handles that already exist.

Unset fields on `OpsImplementationConfig` remain `None`, and
`resolve_op(op, variant, None)` / `VeomniOp(op, variant, None)` selects the
host default for that `(op, variant)` from `OP_REGISTRY.default_impl`: the
first `default=True` row on the active device whose hardware and software
requirements are satisfied, or explicitly `eager` when no default candidate
is available. Registration order only determines priority among available
default candidates. An explicit implementation never falls back when its
requirements are unmet. For example, `rms_norm/unweighted` has no Ascend kernel,
so an unset `rms_norm_implementation` resolves `rms_norm/standard` to
`torch_npu` and `rms_norm/unweighted` to `eager` on NPU.

For compatibility, consumers pass `resolve_op_impl(field, variant=...)`
for variants with explicit-value aliases. In particular, an explicit
`rms_norm_implementation='torch_npu'` (or the deprecated `npu`) still maps to
`eager` for `unweighted`, while weighted variants retain `torch_npu`.

Full RoPE's eager and Liger implementations store fixed cos/sin tables in the
query dtype for backward. Forward computation still receives the original
tables. This keeps saved metadata stable when FSDP2 casts vision tables on the
first forward but non-reentrant checkpoint recomputation sees FP32 tables.
Trainable tables continue through the eager path with table gradients.

The registry wrapper is the canonical tensor contract for an op variant;
it is not a collection of model-specific adapters. Transformations that vary
by consumer stay with that consumer. For example:

- causal shifting and sequence-parallel loss reduction live in
  `models/loss_utils/cross_entropy_loss.py`;
- concatenating per-layer router logits and applying attention masks live in
  `models/loss_utils/load_balancing_loss.py`;
- generated model patches construct the appropriate variant and translate
  model-owned parameters into its tensor contract.

## Compound ops

A compound raw op must call another row's raw `forward`/`backward`, not its
autograd wrapper. `compound.py` provides `resolve_inner_op`, `append_inner`,
and `take_inner` so nested `SavedState` tensors and metadata can be flattened
into the outer custom-autograd state.

## Process-wide integrations

The `sdpa` and `veomni_sdpa` attention rows accept ordinary dense attention
masks but do not expose a packed/varlen API, following the public
[`torch.nn.functional.scaled_dot_product_attention` signature](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
which has `attn_mask` but no `cu_seqlens` arguments. `packed_causal_mask`
composes the segments into a dense block-diagonal mask for these
implementations. Their attention calls and the VeOmni SDPA mask builder drop
cumulative-length and varlen maximum-length metadata when it describes at most
one segment per batch row, or when the call carries a 3-D/4-D dense mask.
Several segments per row without such a mask raise, because SDPA would attend
across samples. Opaque dense masks are not inspected to infer whether they
encode sequence boundaries.

The VeOmni SDPA mask builder only honors causal mask-elision hints for the
canonical causal predicate with equal Q/K lengths and offsets. Bidirectional
elision requires the canonical bidirectional predicate and an explicit hint.
Sliding-window and custom predicates always retain an explicit mask, including
when callers enable skip hints. Transformers still checks padding before any
permitted elision. Eager shape masks always request an explicit mask.

`install.py` contains idempotent process-wide integration only. Currently it
registers VeOmni attention names and mask builders on Transformers registries
and patches the Transformers hub-kernel loader for local FlashAttention
implementations. It is skipped when `MODELING_BACKEND=hf`.

`batch_invariant/` is deliberately separate from `kernels/`: it temporarily
patches ATen implementations through `set_batch_invariant_mode(...)` and is not
selected through `VeomniOp`.

## Tests and further documentation

- Registry and generated-autograd contract: `tests/ops/base/test_op_entry.py`
- Per-family math and hardware behavior: `tests/ops/<family>/`
- Model-facing integration and helpers: `tests/models/`
- User-facing selection and lifecycle: `docs/design/op_selection.md`

When adding a row, test its numerical contract, registration, hardware
requirement, and optional-package requirements. When adding model-specific
argument policy, test it in `tests/models` rather than duplicating it in
the raw-kernel suite.
