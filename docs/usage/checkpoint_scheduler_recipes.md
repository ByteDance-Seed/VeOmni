# Checkpointable custom learning-rate schedules

`LambdaLR.state_dict()` does not serialize the Python function that computes
the schedule. A `MultiOptimizer` checkpoint therefore needs a stable recipe
for each `LambdaLR` leaf in addition to its ordinary scheduler state. VeOmni's
constant, linear and cosine factories register their recipes automatically.

For a custom schedule, the application owns the function and its recipe. Build
and register each leaf scheduler before the first checkpoint save or load, and
repeat that construction on every participating rank and in every fresh resume
process. A `MultiLRScheduler` must use the same leaf names as its optimizer:

```python
from torch.optim.lr_scheduler import LambdaLR

from veomni.optim.lr_scheduler import MultiLRScheduler, register_checkpoint_scheduler


def build_warmup(leaf_optimizer, warmup_steps):
    def factor(step):
        return min(1.0, step / max(1, warmup_steps))

    scheduler = LambdaLR(leaf_optimizer, lr_lambda=factor)
    return register_checkpoint_scheduler(
        scheduler,
        name="my_project.linear_warmup",
        version=1,
        parameters={"warmup_steps": warmup_steps},
    )


lr_scheduler = MultiLRScheduler({
    name: build_warmup(leaf, warmup_steps=100)
    for name, leaf in optimizer.optimizers_dict.items()
})
```

Keep the returned scheduler alive with the optimizer. `name` must be a
nonempty stable string, `version` a positive integer, and `parameters` a
dictionary of finite JSON-serializable values. Include every value that can
change the schedule, including per-leaf differences. Increment the version
when the implementation's behavior changes. The application must use the
same deterministic implementation and recipe for the original and resumed
run; avoid hidden state such as environment values or mutable closure data.
Registration records the caller's assertion and does not inspect or sandbox
the callable.

The owner-aware optimizer checkpoint records the registered leaf identities:
[`_scheduler_layout`](../../veomni/checkpoint/optimizer_resume_layout.py#L125)
reads each leaf's registered recipe during layout construction. During save,
[`ModelCheckpointManager._extra_state`](../../veomni/models/checkpoint/manager.py#L271)
records the scheduler recipe alongside its state. During load, the
[recipe compatibility check](../../veomni/models/checkpoint/manager.py#L389)
compares it with the newly registered recipe. Recovery rejects missing or
different recipes instead of silently attaching a new function to old
optimizer state. Scheduler step, optimizer groups and trainer checkpoint
generation are checked separately.
An unregistered custom `LambdaLR` or an older callable-hash-only checkpoint
needs an explicitly verified migration or a new weights-only run.
