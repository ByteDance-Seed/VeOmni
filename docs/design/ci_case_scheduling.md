# Case-level accelerator test scheduling

The CI scheduler assigns parameterized pytest cases to disjoint accelerator
subsets using each case's launch configuration. Unit and e2e tests share the
same planning and scheduling core; GPU and NPU adapters provide device discovery
and process visibility.

For commands, module responsibilities and result files, see the
[accelerator README](../../scripts/ci/accelerator/README.md).

Independent training launches use fresh distributed workers and process groups.
Scheduling preserves numerical tolerances, worker world sizes, SP/EP/CP/HSDP
configurations, and test data.

## One source of truth for resources

Launchers define `nproc`, `world_size`, `cp_size`, or a resolved
`ParallelConfig`. Expose that configuration as a collection-time `LaunchSpec`
and reuse it to launch the actual worker processes. Do not add a separate
per-case device-count table or a `devices=4` marker alongside `nproc=4`.

A `CasePlan` contains one or more launch specifications plus dependencies. On a
single node, the accelerator reservation is the peak number of distinct cards
required by the case: the maximum for sequential launches, and the sum for
launches that run concurrently. A one-card reference followed by two-card FSDP
needs two cards; concurrent four-card modes need eight. Internal suballocation
must stay inside the case's lease rather than acquire the same resources twice.

CPU process count is separate from accelerator count. Four Gloo CPU workers use
zero accelerator cards. Global `world_size` is not automatically a local card
count when multiple nodes are involved; the scheduler supports one node
and must reject unsupported multi-node plans explicitly.

Resolve hardware-dependent defaults once against the runner's complete owned
slice. For example, `min(available_devices, 4)` resolves to four on an eight-card
runner even when only two cards happen to be free. Wait for four cards instead
of changing the test to two. Port selection and output-directory creation
happen at execution time, not in the pure plan resolver.

## Default launch plan

| Test declaration | On a GPU/NPU runner | On a CPU-only runner |
|---|---|---|
| No explicit launch plan | One card of the selected accelerator backend | CPU, zero accelerator cards |
| Explicit CPU plan | CPU, zero accelerator cards | CPU, zero accelerator cards |
| Explicit accelerator launch spec | Its resolved local `nproc` | Preserve the established unsupported-hardware skip |

Defaulting to CPU on an accelerator runner is unsafe: ordinary tensor tests may
select the CPU path, or accelerator tests may skip and make coverage look
successful. A default one-card plan preserves the selected accelerator backend.
A pure CPU test may keep the default for convenience, at the cost of reserving
one card, or declare a CPU plan to allow more concurrency.

The default is a runnable single-card contract, not an estimate. Every
multi-card test must expose its actual launch configuration before scheduling.
A missing or undersized plan must produce a resource-plan error rather than
silently shrinking the launch, switching to Gloo, or skipping due to a smaller
visible device set. Compare worker collection and outcomes with the inventory
collected against the complete runner slice to catch such losses.

`pytest.mark.skipif(device_count < N)` markers express eligibility,
not exact resource reservations. For example,
[the CP case](../../tests/parallel/context_parallel/test_deepseek_v4_cp.py#L920)
checks a two-device minimum but launches either two or four workers according
to its parameter. The
[launcher defaults](../../tests/tools/launch_utils.py#L119) and
[training command resolution](../../tests/tools/training_utils.py#L228) provide
shared plan inputs rather than a second quota definition.

## Writing new tests

- Place a test inside an existing CI collection scope, or explicitly add its
  selection to the suite. Suite membership is separate from resource planning;
  see [registration rules](../testing.md#registering-tests-in-ci).
- A single-card test may use the default plan. State CPU-only intent when the
  test must not allocate accelerator memory.
- A multi-card test uses a common launcher and a plan derived from its existing
  parameters/configuration. A plan provider can be inherited by a module,
  class, or fixture family; adding parameter combinations must not require
  updating a scheduler table.
- A case with several launches exposes all launch specs and their sequential
  or concurrent relationships. Keep comparison and resume dependencies intact.
- Do not choose a smaller `world_size` from the leased device count. The lease
  is chosen from the plan, not the other way around.
- Use case-scoped output/rendezvous directories and the common port allocator.
  Shared fixture caches must be immutable with locking, or isolated per case;
  one worker must not delete a dataset while another reads it. Keep temporary
  directory paths short enough for multiprocessing Unix sockets.
- Raw `torchrun`, `mp.spawn`, or shell launchers must participate in plan
  validation. Unsupported entry points must be reported explicitly;
  do not pretend to infer arbitrary Python execution from skip markers.

The extension point is a pure plan provider that receives collected parameters
and immutable runner capabilities. Collection and runtime validation consume
its result.

## Test API

Use the `provider` keyword on `pytest.mark.launch_plan`. A provider is either a
[LaunchSpec](../../scripts/ci/accelerator/launch_plan.py#L30), a
[CasePlan](../../scripts/ci/accelerator/launch_plan.py#L48), or a callable accepting a
[PlanContext](../../scripts/ci/accelerator/launch_plan.py#L81) and returning one of those
values. The planner calls it through
[resolve_plan](../../scripts/ci/accelerator/pytest_plugin.py#L22). Using a keyword prevents pytest from interpreting a
callable provider as the test function being decorated.

A fixed distributed launch keeps its process count in one place:

```python
import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import torchrun

PARALLEL = LaunchSpec(nproc=4)

@pytest.mark.launch_plan(provider=PARALLEL)
def test_distributed_parity():
    torchrun(check_parity, world_size=PARALLEL.nproc)
```

A parameterized launch uses the same parameter for planning and execution:

```python
from scripts.ci.accelerator.launch_plan import from_parameter

@pytest.mark.launch_plan(provider=from_parameter("cp_size"))
@pytest.mark.parametrize("cp_size", [2, 4])
def test_context_parallel(cp_size):
    mp.spawn(check_context_parallel, args=(cp_size,), nprocs=cp_size)
```

For CPU-only tests, use `LaunchSpec(device="cpu")`. For CPU distributed tests,
also set their process count, for example `LaunchSpec(nproc=4, device="cpu")`;
those four Gloo workers reserve zero accelerator cards. A module can set
`pytestmark = pytest.mark.launch_plan(provider=...)`, and a class can carry the
same marker. A function-level marker takes precedence. `SequenceParallelTest`
derives its plan directly from the instance's `world_size` property.

`CasePlan.sequential(first, second)` reserves the larger launch;
`CasePlan.concurrent(first, second)` reserves their sum. General plans use
`CasePlan(groups=((first, second), (third,)))`: groups describe sequential phases,
and launches inside a group may execute concurrently. Plans describe resource
requirements; the test's launch code still owns execution and comparisons.

For hardware-dependent configuration, use `context.capabilities` in a provider
and `runner_device_count()` in the corresponding execution helper. Both refer
to the complete runner slice. Use the accelerator runtime's device count for
indexing the worker's actual visible devices.

## Collection and execution

Collect the CI selections with the runner's complete device visibility
and retain parameterized node IDs, eligibility, and resolved launch plans. Do
not execute fixtures or training just to discover resource requirements.

Resolve every multi-device case through its shared launch configuration and
verify the inventory before dispatch. Missing launch plans must not introduce
silent single-card execution.

Cases already skipped in the full-visibility inventory keep their declared
plans and card requirements. If such a request exceeds the runner's capacity,
place it in a skipped-only batch using the complete owned slice so worker
imports retain the same device visibility. Runnable cases must never receive
a smaller reservation than their declared requirement.

Scheduling operates on case records, but launching thousands of fresh pytest
interpreters would add substantial import and fixture cost. Automatically batch
compatible cases with the same resource requirements and fixture affinity,
using fixed weights of 60 seconds per multi-card case and 2 seconds per
single-card or CPU case. The default batch target is 120 estimated seconds,
with at most 64 cases. These weights control batching only; they are not
runtime predictions or priorities and do not bound individual case runtimes.
Execution timeouts bound stalled workers. A batch can reuse
the pytest interpreter as an ordinary module invocation does; distributed
subprocesses still start fresh for each original launch.

Stably sort batches by required card count, largest first; preserve collection
order among equal requests. Scan the queue and dispatch fitting batches onto
disjoint devices, letting smaller jobs fill otherwise idle space. When the
accelerator queue head has waited 120 seconds, the next dispatch check stops
admitting later accelerator jobs and lets active batches drain until that head
fits. CPU work has a separate concurrency limit. No historical timings are
required or consumed. Measured case timings remain available as reporting
artifacts. Results stay keyed by node ID, independent of completion order.

Each worker receives its visible-device mapping and a unique rendezvous port
before importing torch or the accelerator runtime. A run-scoped, locked port
registry prevents reuse while another launch is preparing to bind. Common launchers validate actual `nproc`, node count, and child device visibility
against the plan and lease. Compare observed launch arguments with the baseline
to verify backend and parallel-topology coverage. Release resources only after scoped
cleanup of the launcher's process group and tracked descendants, including
when its group leader has already exited. Failures stop new dispatches by default; the
measurement mode may continue while recording all failures.

## GPU/NPU boundary

The resource queue, plan model, batching, inventory reconciliation, and result
format are backend-neutral. Device discovery, permitted physical IDs, child
visibility, logical-rank mapping, and backend-specific cleanup belong to a
small accelerator adapter. Use the
[device helpers](../../veomni/utils/device.py#L41) for CUDA/NPU and NCCL/HCCL
selection.

The adapter may allocate only from the runner's owned slice. NPU
workflows select either cards 0-7 or 8-15 on a sixteen-card host; scheduling
must not reach into the other runner's slice. Validate the Ascend visible-device
mapping and HCCL communicator/port behavior on hardware. Never run a global
process sweep while concurrent batches are active.

Keep Ascend's process-local device indices distinct from the selected host
IDs: a process restricted to a host device must address it as logical device
zero. The [CANN visibility reference](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0028.html)
describes this remapping and requires IDs in ascending order. Set visibility
before starting the child process. The pinned torch-npu line
[caches its device count after initialization](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/__init__.py#L467),
so changing environment variables inside an initialized worker is insufficient.
Explicit CPU plans also need an NPU-aware availability check; an empty Ascend
visibility string must not be assumed to behave exactly like CUDA's empty
visibility string without checking the installed runtime.

Both unit and e2e suites consume the same planner/scheduler. Suite selection
defines which cases run, and launch plans define their resource requirements.
Hardware and conditional eligibility remain part of test collection.

## Validation and results

- Compare the complete baseline and candidate node-ID/outcome multisets, with
  only explicitly documented duplicate removals and added scheduler tests.
- No new resource-induced skips, changed accelerator backends, or reduced
  SP/EP/CP/HSDP/world-size coverage.
- Check that every observed distributed launch matches its resolved plan.
- Exercise defaults, parameterized plans, CPU workers, sequential/concurrent
  case peaks, mismatches, starvation prevention, fixture isolation, and cleanup.
- Run the full unit/e2e selections, including conditional coverage. Preserve
  per-case timings, actual wall time, device assignments, logs/JUnit, source
  fingerprints, and cache/environment limits.
- Run serial and concurrent GPU/NPU adapter checks on their respective hardware;
  record any NPU validation gap explicitly.
