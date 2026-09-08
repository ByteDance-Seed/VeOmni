# `veomni.kernels`

`veomni.kernels` owns VeOmni's tensor-level kernel registry and kernel
implementations. Model-specific input normalization, Hugging Face-compatible
signatures, and loss policy belong in `veomni.models_kernel`.

Importing `veomni.kernels` registers every built-in kernel family and applies
the process-wide attention integration from `install.py`.

## Layout

```text
veomni/kernels/
├── __init__.py          Register built-in families and apply global integrations
├── registry.py          KernelEntry, register_kernel, resolve_kernel, VeomniKernel
├── requirement.py       CUDA, NPU, and MLU availability requirements
├── compound.py          Saved-state helpers for kernels that call other raw kernels
├── config.py            Installed kernel-selection config read during model construction
├── install.py           Idempotent process-wide attention integration
├── batch_invariant/     Opt-in ATen patch; not a registered kernel family
└── _kernels/            Tensor-level implementations and registrations
```

Python modules maintained by VeOmni document every module, class, and
callable. Directories named `vendor/` mirror external implementations and
retain their upstream source layout and documentation style.

Registered rows use the identity:

```text
(kernel, variant, implementation, device)
```

Callers select the public `(kernel, variant, implementation)` triple. The
registry derives `device` from the row's requirement and resolves the current
device first, followed by a device-agnostic row.

## Built-in families

| Kernel | Variants | Implementations |
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
from veomni.kernels import KERNEL_REGISTRY

KERNEL_REGISTRY.list_registered("rms_norm", "standard")
KERNEL_REGISTRY.list_available("rms_norm", "standard")
```

`list_registered` includes every known implementation. `list_available`
filters those rows using the current device and hardware requirement.

## Registering a kernel

Each row provides either:

- a raw `forward` and `backward` pair, from which the registry generates a
  `torch.autograd.Function` wrapper; or
- one opaque `wrapper`, for eager PyTorch or a library API that already owns
  its autograd behavior.

Do not provide both forms for one row.

```python
from veomni.kernels import register_kernel
from veomni.kernels.requirement import CudaKernelRequirement

register_kernel(
    "example",
    "standard",
    "eager",
    wrapper=eager_example,
)

register_kernel(
    "example",
    "standard",
    "triton",
    forward=triton_forward,
    backward=triton_backward,
    requirement=CudaKernelRequirement(min_cc=80),
)
```

For a raw pair, `forward` returns `(output, SavedState)`, and `backward`
returns one gradient entry for every positional tensor passed to the generated
wrapper. Tensor inputs must therefore be positional; non-tensor attributes
must be keyword arguments.

```python
from veomni.kernels.registry import SavedState


def raw_forward(x, *, scale):
    return x * scale, SavedState((), scale)


def raw_backward(grad_output, saved):
    return (grad_output * saved.metadata,)
```

An explicit non-eager selection never silently falls back. Unknown rows raise
`KeyError`; rows for the wrong device or unmet hardware requirements raise
`RuntimeError`.

## Calling a kernel from modeling

Models construct a local handle once and call it directly:

```python
from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import resolve_kernel_impl


self.veomni_rms_norm = VeomniKernel(
    "rms_norm",
    "standard",
    resolve_kernel_impl("rms_norm_implementation"),
)

hidden_states = self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)
```

`VeomniKernel` resolves its row at construction, is interned by the public
triple, and always calls the row's wrapper. `models_kernel.build_foundation_model`
installs the `OpsImplementationConfig` object in `kernels/config.py` before
constructing the model.

The registry wrapper is the canonical tensor contract for a kernel variant;
it is not a collection of model-specific adapters. Transformations that vary
by consumer stay with that consumer. For example:

- causal shifting and sequence-parallel loss reduction live in
  `models_kernel/loss_utils/cross_entropy_loss.py`;
- concatenating per-layer router logits and applying attention masks live in
  `models_kernel/loss_utils/load_balancing_loss.py`;
- generated model patches construct the appropriate variant and translate
  model-owned parameters into its tensor contract.

## Compound kernels

A compound raw kernel must call another row's raw `forward`/`backward`, not its
autograd wrapper. `compound.py` provides `resolve_inner_kernel`, `append_inner`,
and `take_inner` so nested `SavedState` tensors and metadata can be flattened
into the outer custom-autograd state.

## Process-wide integrations

`install.py` contains idempotent process-wide integration only. Currently it
registers VeOmni attention names and mask builders on Transformers registries
and patches the Transformers hub-kernel loader for local FlashAttention
implementations. It is skipped when `MODELING_BACKEND=hf`.

`batch_invariant/` is deliberately separate from `_kernels`: it temporarily
patches ATen implementations through `set_batch_invariant_mode(...)` and is not
selected through `VeomniKernel`.

## Tests and further documentation

- Registry and generated-autograd contract: `tests/kernels/base/test_kernel_entry.py`
- Per-family math and hardware behavior: `tests/kernels/<family>/`
- Model-facing integration and helpers: `tests/models_kernel/`
- User-facing selection and lifecycle: `docs/design/kernel_selection.md`

When adding a row, test both its numerical contract and its registration or
hardware requirement. When adding model-specific argument policy, test it in
`tests/models_kernel` rather than duplicating it in the raw-kernel suite.
