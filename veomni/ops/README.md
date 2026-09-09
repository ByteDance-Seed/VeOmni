# `veomni.ops`

`veomni.ops` owns VeOmni's tensor-level operation registry and concrete kernel
implementations. Model-specific input normalization, Hugging Face-compatible
signatures, and loss policy belong in `veomni.models_kernel`.

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
device first, followed by a device-agnostic row.

## Built-in families

| Op | Variants | Implementations |
|---|---|---|
| `attention` | `standard` | `eager`, `sdpa`, FlashAttention/FlexAttention/MagiAttention/SageAttention names, `native-sparse` |
| `async_ulysses_qkv`, `async_ulysses_o` | `standard`, `dit` | `eager` orchestration |
| `rms_norm` | `standard`, `unweighted`, `qwen3_5` | `eager`, `liger_kernel`, `npu`, and `triton` where supported |
| `rope` | `full`, `partial`, `deepseek_v4`, `wan` | `eager`, `liger_kernel`, `npu`, and `triton` where supported |
| `rope_vision` | `full` | `eager`, `npu` |
| `swiglu_mlp` | `standard` | `eager`, `liger_kernel` |
| `moe_experts` | `standard`, `gpt_oss` | `eager`, `fused_triton`, `fused_quack`, `fused_npu`, `fused_mlu` as supported by the variant |
| `moe_experts_lora` | `independent`, `shared` | `eager`, `fused_triton`, `fused_npu` |
| `cross_entropy_loss` | `standard` | `eager`, `chunk_loss`, `liger_kernel` |
| `load_balancing_loss` | `standard` | `eager`, `triton` |
| `rms_norm_gated`, `causal_conv1d`, `chunk_gated_delta_rule` | `standard` | `eager`, `fla`, `flash_qla`, `npu`, `npu_ascendc` as supported by the family |
| `dsa_indexer`, `dsa_attention` | `deepseek_v4`, `glm` | `eager`, `tilelang`, `cudnn`, or `flashmla_cudnn` by variant |
| `mhc` | `pre`, `post`, `head` | `eager`, `tilelang` |

The registry is the source of truth for the exact rows available in a given
revision:

```python
from veomni.ops import OP_REGISTRY

OP_REGISTRY.list_registered("rms_norm", "standard")
OP_REGISTRY.list_available("rms_norm", "standard")
```

`list_registered` includes every known implementation. `list_available`
filters those rows using the current device and hardware requirement.

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
    wrapper=eager_example,
)

register_op(
    "example",
    "standard",
    "triton",
    forward=triton_forward,
    backward=triton_backward,
    requirement=GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=80),)),
)
```

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
`KeyError`; rows for the wrong device or unmet hardware requirements raise
`RuntimeError`.

## Calling an op from modeling

Models construct a local handle once and call it directly:

```python
from veomni.ops import VeomniOp
from veomni.models_kernel.utils.op_utils import resolve_op_impl


self.veomni_rms_norm = VeomniOp(
    "rms_norm",
    "standard",
    resolve_op_impl("rms_norm_implementation"),
)

hidden_states = self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)
```

`VeomniOp` resolves its row at construction, is interned by the public
triple, and always calls the row's wrapper. `models_kernel.build_foundation_model`
installs the `OpsImplementationConfig` object in `ops/config.py` before
constructing the model.

The registry wrapper is the canonical tensor contract for an op variant;
it is not a collection of model-specific adapters. Transformations that vary
by consumer stay with that consumer. For example:

- causal shifting and sequence-parallel loss reduction live in
  `models_kernel/loss_utils/cross_entropy_loss.py`;
- concatenating per-layer router logits and applying attention masks live in
  `models_kernel/loss_utils/load_balancing_loss.py`;
- generated model patches construct the appropriate variant and translate
  model-owned parameters into its tensor contract.

## Compound ops

A compound raw op must call another row's raw `forward`/`backward`, not its
autograd wrapper. `compound.py` provides `resolve_inner_op`, `append_inner`,
and `take_inner` so nested `SavedState` tensors and metadata can be flattened
into the outer custom-autograd state.

## Process-wide integrations

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
- Model-facing integration and helpers: `tests/models_kernel/`
- User-facing selection and lifecycle: `docs/design/kernel_selection.md`

When adding a row, test both its numerical contract and its registration or
hardware requirement. When adding model-specific argument policy, test it in
`tests/models_kernel` rather than duplicating it in the raw-kernel suite.
