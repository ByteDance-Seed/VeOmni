# Rank-local AdamW checkpoint adapters

`register_rank_local_adamw` is an internal extension point for an AdamW-like
optimizer whose master weights or moments must retain their own dtype or CPU
placement when restored through `MultiOptimizer` and `DistributedCheckpointer`.
It is not a generic optimizer plugin, a security boundary, or a conversion tool
for an incompatible checkpoint. Register trusted code only.

## Registration and ownership

```python
from contextlib import nullcontext

import torch

from veomni.optim.checkpoint_adapter import register_rank_local_adamw

# Illustrative callbacks for an AdamW-like optimizer that already implements
# precision-preserving state loading; MyAdamW is the application's own class.
register_rank_local_adamw(
    MyAdamW,
    context_factory=lambda optimizer: nullcontext(),
    algorithm_identity=lambda optimizer: ("my-adamw-v1", str(optimizer.moment_dtype)),
    state_dtype=lambda optimizer, param, field: (
        torch.float32 if field in ("step", "master_weight") else optimizer.moment_dtype
    ),
    state_is_offloaded=lambda optimizer, param: optimizer.is_offloaded(param),
)
```

The registry is process-local and matches `type(optimizer)` **exactly**;
register each supported concrete optimizer type before saving or loading on
every rank. Duplicate registration raises an error. Registration does not make
an arbitrary optimizer compatible or automatically preserve its precision: the
optimizer and its adapter own the correctness of their serialization and load
behavior. The in-tree checkpoint adapter tests provide an illustrative
registered implementation.

An initialized parameter entry must contain exactly `step`, `master_weight`,
`exp_avg`, and `exp_avg_sq`. The step is a rank-local plain scalar tensor, not
a scalar DTensor; the other fields have the parameter's shape. A parameter with
no optimizer state must stay absent, rather than receiving synthetic moments
or a synthetic step. The current adapter contract does not support additional
per-parameter fields.

## Callback contract

| Callback | Signature and result | Responsibility |
| --- | --- | --- |
| `context_factory` | `context_factory(optimizer) -> context manager` | Entered while the framework obtains or applies that leaf optimizer's state. It may temporarily select an optimizer's supported serialization mode, but must expose the live initialized fields and group settings without mutating training state. On restore, it must allow the optimizer's loader to install the saved precision and placement without an implicit parameter-dtype cast. `nullcontext()` is sufficient only when the optimizer already has that behavior. |
| `algorithm_identity` | `algorithm_identity(optimizer) -> immutable value` | Describes non-group settings that change updates or restore semantics. Use `None`, booleans, integers, floats, strings, or tuples recursively composed of those types; encode dtypes as strings. Include an algorithm/schema version and all such settings. The checkpoint records this value and rejects a different identity on resume. Group settings and scheduler identity are checked separately. |
| `state_dtype` | `state_dtype(optimizer, param, field) -> torch.dtype` | Returns the expected stored dtype for each of `step`, `master_weight`, `exp_avg`, and `exp_avg_sq`, including when the state is not initialized yet. `step` describes the scalar dtype, not the moment dtype. The framework records the policy and validates existing and restored tensors against it; this callback does not cast tensors. |
| `state_is_offloaded` | `state_is_offloaded(optimizer, param) -> bool` | Reports whether that parameter's master and moment state belongs on CPU. The per-parameter policy is part of checkpoint identity and selects CPU versus parameter-device allocation during restoration. The scalar step follows the optimizer group's capturable/fused placement rules separately. |

Callbacks must be deterministic for the same training configuration and
stable across the save and load. `optimizer` is the live concrete leaf, and
`param` is a parameter owned by that leaf, possibly a DTensor. A fresh process
constructs new optimizer and parameter objects; callbacks must reconstruct the
same policies for them. Parameter ownership, optimizer group
membership, scheduler identity, state dtypes and offload selection must match
the saved checkpoint. Change these only with an explicitly verified migration.

### Save and load call sites

`optimizer_resume_layout._local_layout` records `algorithm_identity`, each
parameter's `state_dtype`, and `state_is_offloaded` in the checkpoint identity;
a mismatch rejects resume before applying state. During save,
`multi_optimizer_checkpoint._validate_source_optimizer_states` uses
`state_dtype` to check initialized fields against their declared precision.
During load, `_prepare_multi_optimizer_saved_states` checks stored field dtypes
and uses `state_is_offloaded` to allocate missing destination state on CPU when
the adapter specifies offload. `context_factory` wraps the optimizer's actual
state read and load; adapter implementations own correct placement and values.

## Framework call sites and restore behavior

1. `MultiOptimizer._sparse_state_dict()` enters the context around PyTorch's
   optimizer-state serialization. Registered adapter implementations must
   preserve live state and serialize all initialized fields and group options
   faithfully. State-empty leaves retain their parameter groups without
   creating optimizer moments.
2. Checkpoint layout construction records the algorithm identity, parameter
   ownership, dtype policy and offload policy. Save validates the source state;
   load checks the checkpoint's identity, field names, dtypes and shapes before
   preparing initialized state for the saved owners. Rank-local ownership and
   the supported parallel geometry are not automatically resharded.
3. `MultiOptimizer._dcp_load_state_dict()` enters the context again while
   PyTorch applies the optimizer state. The trusted adapter must restore every
   validated field without changing its value or dtype. An optimizer may
   implement its own precision-preserving loader; the adapter callbacks alone
   do not repair a lossy one.

The currently supported exact-precision training-checkpoint path requires
`MultiOptimizer` and synchronous `DistributedCheckpointer` save/load.
`MultiOptimizer.load_state_dict()` and raw DCP `OptimizerState` reject
unproven recovery; an individual leaf's direct `load_state_dict()` cannot prove
the MultiOptimizer group, scheduler and generation identity. Unverified older
group layouts and changing the rank-local layout
require an explicit verified migration or fail closed. See
[MultiOptimizer recovery identity](../usage/checkpoint.md#multioptimizer-recovery-identity)
for the complete recovery constraints. Test an extension with initialized and
unused parameters, save, fresh-process restore, and the next update at its
actual dtype and placement; an adapter registration or a single forward pass
is not an acceptance test.
