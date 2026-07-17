# Local Parallel State Registry and Scoping

VeOmni stores distributed topology in `ParallelState`. A process can register
multiple named states and temporarily select the state used by APIs that read
the ambient parallel context. This supports composite models whose modules may
use different sequence-parallel groups while preserving the simple
`get_parallel_state()` interface for ordinary trainers.

## Public API

| API | Purpose |
|-----|---------|
| `init_parallel_state(..., name="base")` | Build a topology, register it under `name`, and establish the first state as the ambient default. |
| `get_parallel_state_by_name(name)` | Retrieve a registered state without changing the ambient state. |
| `use_parallel_state(name_or_state)` | Temporarily make a registered name or `ParallelState` object ambient, then restore the previous state on exit. |
| `get_parallel_state()` | Return the current ambient state; before initialization it returns a single-process state and logs a warning. |
| `clear_parallel_state()` | Clear the ambient state, topology cache, and named registry after distributed teardown. |

`init_parallel_state` maintains two independent mappings:

- The **named registry** maps logical module names to states. Registering an
  existing name logs a warning and returns the existing state.
- The **topology cache** reuses a state when every topology-defining argument
  matches. Different names can therefore refer to the same state without
  creating duplicate process groups.

## Basic usage

```python
from veomni.distributed.parallel_state import (
    get_parallel_state_by_name,
    init_parallel_state,
    use_parallel_state,
)

init_parallel_state(
    dp_size=4,
    dp_shard_size=4,
    ulysses_size=2,
    name="base",
)

base_state = get_parallel_state_by_name("base")

with use_parallel_state("base"):
    output = model(input_ids)
```

The context manager accepts either the registered name or the state object.
It always restores the previous ambient state, including when the scoped
operation raises.

## Multiple module topologies

Register each logical module on every rank, in the same order, before its first
scoped operation. For an eight-rank process group:

```python
init_parallel_state(dp_size=4, dp_shard_size=4, ulysses_size=2, name="thinker")
init_parallel_state(dp_size=8, dp_shard_size=8, ulysses_size=1, name="talker")

with use_parallel_state("thinker"):
    thinker_output = thinker(batch)

with use_parallel_state("talker"):
    talker_output = talker(thinker_output)
```

Sequence-parallel communication helpers resolve their process groups from the
current ambient state. A sequence-parallel state must therefore be created by
`init_parallel_state`; constructing a meshless `ParallelState` with SP enabled
raises an error.

## Trainer lifecycle

Current built-in trainers register the main topology as `"base"` during
`BaseTrainer._setup()`. Model, dataloader, optimizer, and scheduler construction
run inside one `use_parallel_state("base")` build scope. At run time, only
operations that depend on ambient groups are scoped: model forward,
post-forward loss handling, backward, and gradient clipping. Callbacks retain
their state explicitly rather than depending on an ambient scope.

When an API accepts an explicit process group, prefer passing the group from
`get_parallel_state_by_name("base")` instead of opening a broader context.

## Teardown

Destroy distributed process groups before clearing the state registry:

```python
import torch.distributed as dist

from veomni.distributed.parallel_state import clear_parallel_state

dist.destroy_process_group()
clear_parallel_state()
```

Clearing prevents a later distributed session from reusing cached device
meshes or process groups that have already been destroyed.
