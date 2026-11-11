# Optimizer Hook (`configure_optimizer`)

`configure_optimizer(optimizer)` is an optional per-module hook that lets a
module adjust the optimizer `ModuleRuntime` just built for it — typically to
change how weight decay applies to a parameter, or to register a step hook.

- Call site: `ModuleRuntime._build_optimizer` in
  [`omni_module_runtime.py`](../../../veomni/models/seed_omni/accelerated/omni_module/omni_module_runtime.py)
- Contract table: [architecture.md §2.1](../design/architecture.md)
- Worked example: [Qwen3 §7.2](../models/qwen3.md)

## When it runs

Immediately after the module's optimizer is built, and **before** the lr
scheduler is constructed and **before** any checkpoint is loaded:

```
_build_optimizer      -> super()._build_optimizer(param_groups)
                      -> module.configure_optimizer(self.optimizer)   # this hook
_build_lr_scheduler   -> reads the final param groups
checkpoint load       -> restores optimizer state into those groups
```

Both orderings matter. Running before the scheduler means the scheduler sees the
final group layout. Running before the checkpoint load means every checkpoint a
run writes has the same layout — a later change would make earlier saves
unloadable.

It is called inside the module's `ParallelState` scope (`with self._scoped():`),
so a hook may read `get_parallel_state()` and see *this module's* mesh rather
than the orchestrator's.

## Contract

| Aspect | Rule |
|--------|------|
| Signature | `configure_optimizer(self, optimizer: torch.optim.Optimizer) -> None` |
| Defined | Optional. Absent means "no adjustment"; `ModuleRuntime` checks for the attribute |
| Called | Once per module, per optimizer construction (so once per run, before resume) |
| May mutate | `optimizer.param_groups` (including adding groups), and `optimizer` itself (e.g. `register_step_pre_hook`) |
| Must not | Rebuild the optimizer, replace the parameters the trainer owns, or assume a specific `lr` |
| Param naming | The optimizer passed in is the module's own; a hook that changes the group count owns that consequence (see below) |

The trainer builds one optimizer per module and aggregates them behind
`MultiOptimizer`; the hook sees only its own module's optimizer. A module with
no trainable parameters never gets one, so the hook does not run.

## Changing the param-group layout

This is the one part of the contract with a checkpoint consequence, so it is
worth stating separately.

`optimizer.param_groups` is part of the optimizer-state key space: a saved state
dict records a state per group, and restoring it against a different group count
is rejected. A hook that splits or merges groups therefore **ties every
checkpoint written by the run to the presence of that hook**. Resuming such a
checkpoint under a run whose module does not perform the same split is not
supported and fails at restore.

Two patterns are in use, and both keep the group count stable across a run:

- **Split a shared group.** Remove the parameter from its group and
  `optimizer.add_param_group({**group, "params": [param], "weight_decay": 0.0})`.
- **Reuse a single-parameter group.** If the parameter already has its own group,
  zero that group's `weight_decay` in place — no new group, no layout change.

Note the asymmetry: the first changes the count, the second does not. Prefer the
second when the module can arrange it, because it leaves resume compatibility
with a run that predates the hook.

## Hyperparameters are not durable

A checkpoint restores param groups from the hyperparameters recorded at save
time. So a hook that edits a group field as a one-off setup step is undone by the
next resume — the field comes back with its saved value.

If the edit must hold across a restore, apply it from a step hook rather than at
construction:

```python
def configure_optimizer(self, optimizer):
    group = _param_group_of(optimizer, self.weight)
    self._row_weight_decay = group["weight_decay"]
    group["weight_decay"] = 0.0
    optimizer.register_step_pre_hook(self._reapply)

def _reapply(self, optimizer, args, kwargs):
    group = _param_group_of(optimizer, self.weight)
    if group is None or self.weight.grad is None:
        return
    group["weight_decay"] = 0.0      # a restore put it back
    # ... apply the intended decay to the rows that should get it
```

The step pre-hook runs before every update, including the first one after a
resume, so the edit survives. `weight.grad is None` is worth guarding on: AdamW
skips parameters without gradients, and so should the hook.

## Relationship to `freeze_model`

`freeze_model()` and `configure_optimizer()` are separate hooks with separate
jobs, and a module may need both:

- `freeze_model()` controls **which parameters get a gradient**
  (`requires_grad`, and masking tricks such as a per-row gradient mask).
- `configure_optimizer()` controls **what the optimizer does outside the
  gradient** — chiefly decoupled weight decay, which `p *= 1 - lr * wd` applies to
  the whole parameter regardless of the gradient.

A frozen parameter has no gradient and AdamW skips it, so weight decay does not
reach it. But a *partially* trainable parameter — a table whose gradient is
masked row-wise — is not skipped: AdamW decays all of it. That case needs
`configure_optimizer` to take the decay over, which is exactly what the
[Qwen3 vision recipe](../models/qwen3.md) does.

## Testing a hook

Three behaviours are worth pinning, because each has a distinct failure mode:

1. **The layout change** — the group count and the `weight_decay` of each group
   after the hook runs.
2. **The step behaviour** — parameters the hook protects are unchanged by a step
   (no gradient and no decay), and parameters it targets do move.
3. **The restore rule** — set the group hyperparameters back to their saved
   values (what a checkpoint load does), run a step, and require the protection to
   hold. Without this, a hook that works on a fresh optimizer can silently stop
   working after a resume.

Distributed wraps need their own case: a hook that slices a parameter by row has
to compute the right offset for an FSDP2-sharded (DTensor) parameter, which a
single-process test on an unsharded tensor cannot see.
