# Migrate from OpSlot to VeomniOp

This note is the breaking-change map from the old OpSlot / `kernel_registry`
surface to the current op registry. Use it when updating modeling, configs, or
out-of-tree kernels. Current authoring and selection details live in
[`op_selection.md`](op_selection.md) and `veomni/ops/README.md`.

## What replaced what

| Old | Current |
| --- | --- |
| `OpSlot`, `VeomniKernel` | `VeomniOp` |
| `KERNEL_REGISTRY`, `kernel_registry.py` | `OP_REGISTRY` in `veomni/ops/registry.py` |
| `register_kernel` / `resolve_kernel` | `register_op` / `resolve_op` |
| `KernelEntry` | `OpEntry` |
| `dispatch.py`, config singleton, process-wide bind | `OpsImplementationConfig` plus instance-local handles |
| `kernels_implementation` | `ops_implementation` |
| `use_non_eager_impl` / `bound_kernel()` | Construct `VeomniOp` in `__init__` and call that handle |
| `veomni.kernels`, `veomni.models_kernel` | `veomni.ops`, `veomni.models` |

Deleted compatibility shims include `Function.apply` factory closures, the
async-Ulysses wrapper allowlist, and global pointers such as
`_fused_moe_forward`. Do not restore them.

## Config

YAML and CLI now use `model.ops_implementation.*`.

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    rms_norm_implementation: liger_kernel
    moe_implementation: fused_triton
    cross_entropy_loss_implementation: liger_kernel
```

`build_foundation_model(..., ops_implementation=...)` installs that object
before model construction. Changing the process config later does not retarget
handles that already exist.

NPU no longer keeps fake hardware aliases. Unweighted RMSNorm and
cross-entropy have no `npu` row. Use `eager` or `chunk_loss` as documented in
[`op_selection.md`](op_selection.md).

## Authors

Register a raw `forward` / `backward` pair or one opaque `wrapper`. The
registry generates the autograd `Function` for a raw pair.

```python
from veomni.ops import register_op

register_op(
    "example",
    "standard",
    "eager",
    description="PyTorch reference implementation of the example operation",
    wrapper=eager_example,
)
```

A raw `forward` returns `(output, SavedState)`. Outputs are
`Tensor | tuple[Tensor, ...]`. Do not hand-write a `Function.apply` closure
and register that.

Compound ops resolve nested rows and call raw `forward` / `backward`. Do not
call `VeomniOp.__call__` inside another custom autograd function.

## Modeling

Construct one local handle per op and always call it.

```python
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl

self.veomni_rms_norm = VeomniOp(
    "rms_norm",
    "standard",
    resolve_op_impl("rms_norm_implementation"),
)
hidden_states = self.veomni_rms_norm(
    hidden_states, self.weight, eps=self.variance_epsilon
)
```

There is no bind phase and no `use_non_eager_impl` fork back to an inlined
Hugging Face body. Attention still installs `veomni_*` names on
`ALL_ATTENTION_FUNCTIONS`, then the patched module stores
`VeomniOp("attention", "standard", config._attn_implementation)`.

Mask builders stay under `veomni/ops/kernels/attention/mask/` and
`veomni/ops/mask.py`. They are not `OP_REGISTRY` rows.

Loss policy that is not tensor-level CE or load-balancing math stays in
`veomni/models/loss_utils/`.

## Callers

Pass `ops_implementation` into `build_foundation_model`. Trainer, LoRA, and
task entry points must not keep `kernels_implementation`. Sequence-parallel
and DiT paths consume the same registry rows. Async Ulysses QKV/O are
registered ops with `standard` and `dit` variants.

## Tests

- Registry and generated autograd: `tests/ops/base/`
- Family numerics and hardware guards: `tests/ops/<family>/`
- Model construction and handle isolation: `tests/models/`
- Stale package-path guards: `tests/special_sanity/test_ops_architecture.py`
